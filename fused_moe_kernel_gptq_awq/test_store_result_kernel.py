import triton
import triton.language as tl
import torch

@triton.jit
def test_store_result_kernel(
    accumulator_ptr,
    # Pointers to matrices
    a_ptr,          # 输入 token 向量
    b_ptr,          # MOE 专家权重矩阵
    c_ptr,          # 输出张量
    b_scale_ptr,    # 每个专家权重的量化 scale
    b_zp_ptr,       # 每个专家权重的量化零点（未使用）
    topk_weights_ptr,  # 每个 token 的 top-k 路由概率（未使用）
    sorted_token_ids_ptr,  # 按专家排序后的 token 索引
    expert_ids_ptr,  # 每个 block 的专家 id
    num_tokens_post_padded_ptr,  # 填充后的 token 数量
    # Matrix dimensions
    N: tl.constexpr,       # 输出特征维度
    K: tl.constexpr,       # 输入特征维度
    EM,                    # 按专家分组后的总 token 块数
    num_valid_tokens,      # 有效 token 数量
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,       # 量化分组大小
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,     # M 维度分块大小
    BLOCK_SIZE_N: tl.constexpr,     # N 维度分块大小
    BLOCK_SIZE_K: tl.constexpr,     # K 维度分块大小
    GROUP_SIZE_M: tl.constexpr,     # M 维度的分组大小
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,             # top-k 路由
    compute_type: tl.constexpr,      # 计算数据类型
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,           # K 维度是否为 BLOCK_SIZE_K 的整数倍
):
    pid = tl.program_id(axis=0)             # 当前程序 id
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)   # M 维度的分块数量
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)    # N 维度的分块数量
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group      # 当前分组 id
    first_pid_m = group_id * GROUP_SIZE_M   # 当前分组的第一个 M 维度分块 id
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # 当前块的M维度ID
    pid_n = (pid % num_pid_in_group) // group_size_m                 # 当前块的N维度ID

    # 加载累加器（模拟原逻辑中的计算结果）
    accumulator = tl.load(accumulator_ptr + 
                         tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + 
                         tl.arange(0, BLOCK_SIZE_N)[None, :])
    
    # 1. 将累加结果转为目标类型（如fp16）
    accumulator = accumulator.to(compute_type)
    
    # 2. 计算输出N维度的偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 3. 加载token索引和mask（模拟原逻辑中的token信息）
     # 当前块的token索引偏移（BLOCK_SIZE_M个连续token）
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # 从sorted_token_ids_ptr加载实际token索引（已按专家分组排序）
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    # 生成token有效mask（仅处理小于num_valid_tokens的真实token）
    token_mask = offs_token < num_valid_tokens

    # 4. 计算输出矩阵的指针（形状：[BLOCK_SIZE_M, BLOCK_SIZE_N]）
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    
    # 5. 生成输出有效mask（只写入有效token和有效N区域）
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    
    # 6. 存储结果
    tl.store(c_ptrs, accumulator, mask=c_mask)


def test_fn():
    test_data = torch.load("62_simple_fused_moe_test_data.pt", map_location="cpu", weights_only=True)

    device = "npu:0"

    A = test_data["A"].to(device)   # 16, 64
    B = test_data["B"].to(device)   # 2, 32, 64
    C = test_data["C"].to(device)   # 16, 2, 32
    B_scale = test_data["B_scale"].to(device)   # 2, 32, 4
    B_zp = test_data["B_zp"].to(device) if test_data["B_zp"] is not None else None  # None
    topk_weights = test_data["topk_weights"].to(device)
    sorted_token_ids = test_data["sorted_token_ids"].to(device)
    expert_ids = test_data["expert_ids"].to(device)
    num_tokens_post_padded = test_data["num_tokens_post_padded"].to(device)

    N = test_data["N"]
    K = test_data["K"]
    EM = test_data["EM"]
    num_valid_tokens = test_data["num_valid_tokens"]

    stride_am = test_data["stride_am"]
    stride_ak = test_data["stride_ak"]
    stride_be = test_data["stride_be"]
    stride_bk = test_data["stride_bk"]
    stride_bn = test_data["stride_bn"]
    stride_cm = test_data["stride_cm"]
    stride_cn = test_data["stride_cn"]
    stride_bse = test_data["stride_bse"]
    stride_bsk = test_data["stride_bsk"]
    stride_bsn = test_data["stride_bsn"]
    stride_bze = test_data["stride_bze"]
    stride_bzk = test_data["stride_bzk"]
    stride_bzn = test_data["stride_bzn"]

    group_size = test_data["group_size"]
    MUL_ROUTED_WEIGHT = test_data["MUL_ROUTED_WEIGHT"]
    top_k = test_data["top_k"]
    compute_type = test_data["compute_type"]
    compute_type = tl.float32
    has_zp = test_data["has_zp"]
    use_int4_w4a16 = test_data["use_int4_w4a16"]
    use_int8_w8a16 = test_data["use_int8_w8a16"]
    even_Ks = test_data["even_Ks"]
    BLOCK_SIZE_M = test_data["BLOCK_SIZE_M"]    # 16
    BLOCK_SIZE_N = test_data["BLOCK_SIZE_N"]    # 16
    BLOCK_SIZE_K = test_data["BLOCK_SIZE_K"]    # 16
    GROUP_SIZE_M = test_data["GROUP_SIZE_M"]    # 1

    print("=" * 50)
    print(f"A.shape={A.shape}, B.shape={B.shape}, C.shape={C.shape}") # A.shape=torch.Size([16, 64]), B.shape=torch.Size([2, 32, 64]), C.shape=torch.Size([16, 2, 32])
    print(f"B_scale.shape={B_scale.shape}") # B_scale.shape=torch.Size([2, 32, 4])
    print(f"B_zp.shape={B_zp.shape if B_zp is not None else None}") # B_zp.shape=None
    print(f"topk_weights.shape={topk_weights.shape}") # topk_weights.shape=torch.Size([16, 2])
    print(f"sorted_token_ids.shape={sorted_token_ids.shape}") # sorted_token_ids.shape=torch.Size([32])
    print(f"expert_ids.shape={expert_ids.shape}") # expert_ids.shape=torch.Size([32])
    print(f"num_tokens_post_padded={num_tokens_post_padded}") # num_tokens_post_padded=32
    print(f"N={N}, K={K}, EM={EM}, num_valid_tokens={num_valid_tokens}") # N=32, K=64, EM=32, num_valid_tokens=32
    # stride_am=64, stride_ak=1, stride_be=2048, stride_bk=1, stride_bn=64
    print(f"stride_am={stride_am}, stride_ak={stride_ak}, stride_be={stride_be}, stride_bk={stride_bk}, stride_bn={stride_bn}") 
    # stride_cm=32, stride_cn=1
    print(f"stride_cm={stride_cm}, stride_cn={stride_cn}")
    # stride_bse=128, stride_bsk=1, stride_bsn=4
    print(f"stride_bse={stride_bse}, stride_bsk={stride_bsk}, stride_bsn={stride_bsn}")
    # stride_bze=0, stride_bzk=0, stride_bzn=0
    print(f"stride_bze={stride_bze}, stride_bzk={stride_bzk}, stride_bzn={stride_bzn}")
    print(f"group_size={group_size}") # 16
    print(f"top_k={top_k}") # 2
    # BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_M=1
    print(f"BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M}") # BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16, GROUP_SIZE_M=1

    print(f"MUL_ROUTED_WEIGHT={MUL_ROUTED_WEIGHT}") # False
    print(f"compute_type={compute_type}") # torch.float32
    print(f"has_zp={has_zp}") # False
    print(f"use_int4_w4a16={use_int4_w4a16}") # False
    print(f"use_int8_w8a16={use_int8_w8a16}") # True
    print(f"even_Ks={even_Ks}") # True

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) *
        triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )   # grid=(4,)

    C_initial = C.clone()
    # 模拟累加器的结果
    accumulator = torch.arange(
        0, BLOCK_SIZE_M * BLOCK_SIZE_N, dtype=torch.float32, device=device
    ).view(BLOCK_SIZE_M, BLOCK_SIZE_N).contiguous()

    print(f"accumulator.shape={accumulator.shape}, accumulator.dtype={accumulator.dtype}") # accumulator.shape=torch.Size([16, 16]), accumulator.dtype=torch.float32
    # print(f"accumulator=\n{accumulator}")

    test_store_result_kernel[grid](
        accumulator,
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        group_size=group_size,
        MUL_ROUTED_WEIGHT=MUL_ROUTED_WEIGHT,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=has_zp,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        even_Ks=even_Ks,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        # # 调试参数
        # DBG_ENABLE=DBG_ENABLE, DBG_PID_M=DBG_PID_M, DBG_PID_N=DBG_PID_N, DBG_K=DBG_K,
        # dbg_a_ptr=dbg_a, dbg_b_raw_ptr=dbg_b_raw, dbg_scale_ptr=dbg_scale,
        # dbg_b_deq_ptr=dbg_b_deq, dbg_acc_ptr=dbg_acc,
    )

    print("=" * 50)
    print(f"Before cast dtypes: A={A.dtype}, B={B.dtype}, B_scale={B_scale.dtype}, C={C.dtype}")
    # A=torch.float32, B=torch.int8, B_scale=torch.float32, C=torch.float32
    print("NPU计算结果：")
    print(f"C矩阵形状: {C.shape}, dtype={C.dtype}")
    print("Kernel计算结果：", C.cpu().tolist())
    


if __name__ == "__main__":
    test_fn()
