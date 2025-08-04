"""
=========== FA Triton Kernel 泛化性测试 ===========
已知 测试 表格: 
  FlashAttentionScore_step64_case_d64_Result.xls 
  FlashAttentionScore_step64+7_case_d64_Result.xls

>> 每一行代表一个测试用例，包含以下字段(共78个)及示例内容：
Group: FlashAttentionScore
Testcase Name: FlashAttentionScore_BNSD_{num} / FlashAttentionScore_BSH_{num}
Enable: disable/onlypref
Level: level0
Network Type: fanhua
B: 1                    #
N1: 24
N2: 3
S1: 64
S2: 64
D: 64
Dtype: bf16
sparse mode: 0
pre tockens: 65536
next tockens: 65536
Layout: BNSD / BSH
PSE: None
pse type: None
Atten mask Dtype: None
Atten mask Shape: None
Padding Mask: None
keep prob: 1
Expect out pricision: 
Expect out err max: 
Expect out err sum: 
Expect out eb: 
Expect dp pricision: 
Actual out pricision: 0
Actual out err max: 0
Actual out err sum: 0
Actual out eb: 0
Actual dp pricision: 0
Actual dp err max: 0
Actual dp err sum: 0
Actual dp eb: 0
Actual dk pricision: 0
Actual dk err max: 0
Actual dk err sum: 0
Actual dk eb: 0
Actual dv pricision: 0
Actual dv err max: 0
Actual dv err sum: 0
Actual dv eb: 0
Actual Memory: 0
Actual kernel time forward: 0.0316
Actual kernel time backward: 0.0979
Actual e2e time forward: 0
Actual e2e time backward: 0
Precision result: Fail
Rmse result: Pass
Rme result: Pass
EB result: 
Performance result: Fail
Memory result: Fail
running status: PASS
Actual kernel time forward transpose: 0.0694
Actual kernel time backward transpose: 0.5471
Actual kernel time forward pad: 0
Actual kernel time backward pad: 0.0000
Actual kernel time forward slice: 0
Actual kernel time backward slice: 0.0000
Actual kernel time forward gpu:
Actual kernel time backward gpu:
BNSD+transpose: 0.1010
>>
任务描述：基于表格数据驱动 test_op_fwd, 完成泛化性测试（精度+性能）。
精度测试：直接 NPU 侧对比 ref_out 和 tri_out 的结果。
性能测试：与 GPU 侧对比，计算 kernel_time 和 e2e_time。
最终目的：泛化性验证 FlashAttentionScore 的精度和性能。
=========== FA Triton Kernel 泛化性测试 ===========
"""

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import os
import time
import pandas as pd
from typing import Dict, Tuple, Optional, Any


# ========== 全局变量和常量 ==========
DEVICE = "npu"
TEST_DATA_DIR = "./test_data"
RESULT_DIR = "./test_results"
os.makedirs(RESULT_DIR, exist_ok=True)

test_results = []  # 全局结果存储
valid_fields = ["Z", "H", "N_CTX", "HEAD_DIM", "causal", "dtype", "BM", "BN", "step", "Group", "Testcase Name", "sparse mode"]
dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

# ========== Triton Kernel 实现（保持不变） ==========
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    # causal = true
    # stage = 1
    # 因果注意力，顾名思义，它在计算时会限制信息的流动，只允许模型看到当前位置及之前的位置
    # 的信息。也就是说，当前位置的输出只能依赖于该位置及其之前的输入，而不能访问当前位置
    # 之后的信息。因果注意力保证了数据的顺序性，避免了“未来信息”的泄露。
    # 但是后面的逻辑也会触发
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # k 之前的版本，随路做转置的版本
    #K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    
    # 修改后不转的版本
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
         # k 之前的版本，随路做转置的版本
        #qk = tl.dot(q, k)
        
        # 修改K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        

        # ------------------------------

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            # p = p.to(tl.float16)
            p = p.to(v.dtype)

        # -------------------------------
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        # k 之前的版本，随路做转置的版本
        #K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, M, Out, sm_scale,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              Z: tl.constexpr, H: tl.constexpr, 
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    # ???, why
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)
    # off_hz = tl.program_id(1) 
    for off_hz in range(0,Z*H):
        off_z = off_hz // H
        off_h = off_hz % H

        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        # block pointers
        # (32, 64)
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            # doesn't matter
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),

            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),

            # doesn't matter
            order=(1, 0),
        )
        # v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        # V_block_ptr = tl.make_block_ptr(
        #     base=V + qvk_offset,
        #     shape=(N_CTX, HEAD_DIM),
        #     strides=(stride_vk, stride_vn),
        #     offsets=(0, 0),
        #     block_shape=(BLOCK_N, HEAD_DIM),
        #     order=v_order,
        # )
        V_block_ptr = tl.make_block_ptr(

            base=V + qvk_offset,

            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),

            offsets=(0, 0),
            # why block_n??
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        
        # k 之前的版本，随路做转置的版本
        #K_block_ptr = tl.make_block_ptr(
        #    base=K + qvk_offset,
        #    shape=(HEAD_DIM, N_CTX),

        #    strides=(stride_kk, stride_kn),
        #    offsets=(0, 0),
        #    block_shape=(HEAD_DIM, BLOCK_N),
        #    order=(0, 1),
        #)
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # initialize offsets

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        # initialize pointer to m and l

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # load scales

        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                            start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
        # stage 2: on-band

        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                            start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + off_hz * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        # tl.static_assert(acc.dtype == tl.float32)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, BM, BN):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        print(f"[debug]o shape: {o.shape}, dtype: {o.dtype}")

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        # if is_hip():
        #     waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        #     extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        grid = (triton.cdiv(q.shape[2], BM),1, 1)
        # (1, 2, 1024)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, M, o, sm_scale, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], N_CTX=q.shape[2],  # why varidic??
            HEAD_DIM=HEAD_DIM_K,  # 64
            BLOCK_M = BM, # 32
            BLOCK_N = BN, # 32
            STAGE=stage,
            debug=True,
            **extra_kern_args)
        # N_CTX=q.shape[2]
        # M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
        # p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        # if causal:
        #     p[:, :, M == 0] = float("-inf")
        # p = torch.softmax(p.float(), dim=-1).half()
        # # p = torch.exp(p)
        # o = torch.matmul(p, v)

        ctx.save_for_backward(q, k, v, o, M)
        # ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        print(f"[debug]o after forward shape: {o.shape}, dtype: {o.dtype}")
        return o


attention = _attention.apply
# ========== Triton Kernel 实现（保持不变） ==========

def extract_test_case_data(
        paths: Dict[str, str],
        extract_map: Dict[str, str],
        new_field: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
    """
    从多个 Excel 文件中提取测试用例数据。
    :param paths: 多个文件路径, 例如 {"file1": "path/to/file1", "file2": "path/to/file2"}
    :param extract_map: 提取字段映射
    :param new_field: 新字段及其值, 例如 {"new_field": "value"}
    :return: 提取的测试用例数据
    Example:
    paths = {
        "64": "FlashAttentionScore_step64_case_d64_Result.xls",
        "7": "FlashAttentionScore_step64+7_case_d64_Result.xls"
    }
    extract_map = {
        "Group": "Group",
        "Testcase Name": "Testcase Name",
        "Level": "Level",
        "Network Type": "Network Type",
        "B": "Z",
        "N1": "H",
        "S1": "N_CTX",
        "D": "HEAD_DIM",
        "Dtype": "dtype",
        "sparse mode": "sparse_mode",
        "Layout": "Layout",
        # 其他需要提取的字段...
    }
    new_field = {
        "BM": 32,
        "BN": 32,
        "causal": False,
    }
    """
    dfs = []
    for key, path in paths.items():
        df = pd.read_excel(path)
        df.insert(0, "step", key)   # 新增文件来源列
        dfs.append(df)
    if not dfs:
        raise ValueError("所有文件加载失败，请检查路径")

    combined_df = pd.concat(dfs, ignore_index=True).fillna("")  # 合并所有 DataFrame

    # 提取并重命名字段
    extract_map["step"] = "step"  # 确保 step 字段也被提取
    missing_cols = [col for col in extract_map.keys() if col not in combined_df.columns]
    if missing_cols:
        raise KeyError(f"缺失列: {missing_cols}")
    extracted_data = combined_df[list(extract_map.keys())].rename(columns=extract_map)

    # 如果有新字段，添加到 DataFrame 中
    if new_field:
        for field, value in new_field.items():
            extracted_data[field] = value
    # 映射数据类型
    if 'dtype' in extracted_data.columns:
        extracted_data['dtype'] = extracted_data['dtype'].map(dtype_map)
    # 确保 step 是首列
    columns = ["step"] + [col for col in extracted_data.columns if col != "step"]
    extracted_data = extracted_data[columns]
    # 展示前10行数据
    # 临时设置显示选项
    pd.set_option('display.max_columns', None)        # 显示所有列
    pd.set_option('display.width', None)              # 自动换行
    pd.set_option('display.max_colwidth', 50)         # 列宽足够
    print("Extracted test cases (head):\n", extracted_data.head(10))
    return extracted_data


def precision_atol_rtol(dtype) -> Tuple[float, float]:
    """
    根据数据类型返回精度的绝对误差和相对误差
    """
    return {
        torch.float16: (1e-3, 1e-2),
        torch.bfloat16: (5e-3, 5e-2),
        torch.float32: (1e-4, 1e-4),
    }.get(dtype, (1e-4, 1e-4))


def compute_errors(ref: torch.Tensor, tri: torch.Tensor) -> Dict[str, float]:
    """
    计算多种误差指标
    """
    diff = (ref - tri).abs()
    return {
        "err max": diff.max().item(),
        "err sum": diff.sum().item(),
        "err mean": diff.mean().item(),
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
    }


# def pytest_sessionfinish(session, exitstatus):
#     """
#     测试会话结束时，保存测试结果到文件
#     """
#     if not test_results:
#         print(">> No test results to save. `test_results` is empty.")
#         return
#     try:
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         result_file = os.path.join(RESULT_DIR, f"test_results_{timestamp}.xlsx")
#         # 保留所有字段，没有字段则填充 ""
#         df = pd.DataFrame(test_results).fillna("")
#         df.to_excel(result_file, index=False)
#         print(f"\n>> 测试完成，结果已保存至 {result_file}")
#         print(f"总计 {len(df)} 个测试用例")
#         print(f"通过: {len(df[df['Precision result'] == 'Pass'])}")
#         print(f"失败: {len(df[df['Precision result'] == 'Fail'])}")
#         print(f"异常: {len(df[df['Precision result'] == 'ERROR'])}")
#     except Exception as e:
#         print(f"保存测试结果时发生错误: {e}")
#         pytest.fail(f"测试结果保存失败: {e}")


# 测试用例生成
def pytest_generate_tests(metafunc):
    """
    pytest hook to generate test cases dynamically
    """
    if 'test_case' in metafunc.fixturenames:
        # 生成测试用例数据
        paths = {
            "64": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64_case_d64_Result.xls"),
            "7": os.path.join(TEST_DATA_DIR, "FlashAttentionScore_step64+7_case_d64_Result.xls")
        }
        extract_map = {
            "Group": "Group",
            "Testcase Name": "Testcase Name",
            # "Enable": "Enable",
            # "Level": "Level",
            # "Network Type": "Network Type",
            "B": "Z",
            "N1": "H",
            "S1": "N_CTX",
            "D": "HEAD_DIM",
            "Dtype": "dtype",
            "sparse mode": "sparse mode",
            "Layout": "Layout",
        }
        new_field = {
            "BM": 32,
            "BN": 32,
            "causal": False,
        }
        # 提取测试数据
        test_data = extract_test_case_data(paths, extract_map, new_field)
        test_cases = [row[valid_fields].to_dict() for _, row in test_data.iterrows()]
        # 确保只对 test_case 参数化一次
        metafunc.parametrize("test_case", test_cases, ids=[f"{case['step']}_{case['Testcase Name']}" for case in test_cases])


def test_op_fwd(test_case:  Dict[str, Any]):
    # Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN, step, group, test_name, sparse_mode = test_case
    Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN, step, group, test_name, sparse_mode = [test_case[k] for k in valid_fields]
    print(f"\nRunning test case: {step}-{test_name} | Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, causal={causal}, dtype={dtype}, BM={BM}, BN={BN}, sparse_mode={sparse_mode}")
    torch.manual_seed(20)
    # 创建输入张量 BNSB
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    sm_scale = 0.5
    try:
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        print(f"[debug]p shape: {p.shape}, dtype: {p.dtype}")

        if causal:
            p[:, :, M==0] = float('-inf')
        p = torch.softmax(p.float(), dim=-1).half().to(v.dtype)
        print(f"[debug]p after softmax shape: {p.shape}, dtype: {p.dtype}")

        ref_out = torch.matmul(p, v)
        print(f"[debug]ref_out shape: {ref_out.shape}, dtype: {ref_out.dtype}")
        # triton kernel
        tri_out = attention(q, k, v, causal, sm_scale, BM, BN)

        print(f"[debug]tri_out shape: {tri_out.shape}, dtype: {tri_out.dtype}")

        atol, rtol = precision_atol_rtol(dtype)         # 误差分析
        errors = compute_errors(ref_out, tri_out)
        passed = torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
        test_results.append({
            "step": step,
            "Group": group,
            "Testcase Name": test_name,
            "B": Z,
            "N1": H,
            "S1": N_CTX,
            "D": HEAD_DIM,
            "Dtype": dtype,
            "sparse mode": sparse_mode,
            "Layout": "BNSD",
            "BM": BM,
            "BN": BN,
            "causal": str(causal),
            "Precision result": "Pass" if passed else "Fail",
            **{f"Actual out {k}": v for k, v in errors.items()},
        })

        assert passed, f"Test failed [{step}-{test_name}]| err_max={errors['err_max']:.2e}, atol={atol}, rtol={rtol}"
    except Exception as e:
        # 捕获异常并记录测试结果
        test_results.append({
            "step": step,
            "Group": group,
            "Testcase Name": test_name,
            "B": Z,
            "N1": H,
            "S1": N_CTX,
            "D": HEAD_DIM,
            "Dtype": dtype,
            "sparse mode": sparse_mode,
            "Layout": "BNSD",
            "BM": BM,
            "BN": BN,
            "causal": causal,
            "Precision result": "Error",
            "Error Message": str(e),
        })
        pytest.fail(f"Test failed with exception [{step}-{test_name}]: {e}")
