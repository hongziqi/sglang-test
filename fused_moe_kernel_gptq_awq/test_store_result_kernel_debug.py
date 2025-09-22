import triton
import triton.language as tl
import torch

@triton.jit
def test_store_result_kernel(
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
    # ---- 调试输出缓冲区（全部视为 row-major 连续）----
    dbg_a_ptr,        # 存储 kernel 加载的 A 数据 (float32)
    dbg_b_raw_ptr,    # 存储 kernel 加载的 B 原始 int8 数据 (float32)
    dbg_scale_ptr,    # 存储 kernel 加载的 scale 数据 (float32)
    dbg_b_deq_ptr,    # 存储反量化后的 B 数据 (float32)
    dbg_offs_token_ptr,  # 存储 kernel 加载的 token 索引 (int64)
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

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)    # 加载padding后的总token数
    # 判断当前 block 是否越界（padding 区域），越界则直接 return
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # 当前块的token索引偏移（BLOCK_SIZE_M个连续token）
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # 从sorted_token_ids_ptr加载实际token索引（已按专家分组排序）
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # [BLOCK_SIZE_M]
    # 生成token有效mask（仅处理小于num_valid_tokens的真实token）
    token_mask = offs_token < num_valid_tokens

    # --------------------------
    # 2. 存储调试变量：offs_token（token 索引）
    # --------------------------
    dbg_offs_token_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    tl.store(dbg_offs_token_ptr + dbg_offs_token_offset, offs_token)

    # 通过 expert_ids_ptr 得到当前 block 的专家 id（off_experts）
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # N维度的块内偏移（当前块在N维度的起始位置）
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N # [BLOCK_SIZE_N]
    # K维度的块内偏移（每次循环处理的K子块）
    offs_k = tl.arange(0, BLOCK_SIZE_K) # [BLOCK_SIZE_K]
    # 输入token的指针（形状：[BLOCK_SIZE_M, BLOCK_SIZE_K]，每个元素对应A矩阵的一个位置）
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )   # [BLOCK_SIZE_M, BLOCK_SIZE_K]

    # === （1.简化kernel 逻辑）仅测int8的量化
    # 专家权重的指针（形状：[BLOCK_SIZE_K, BLOCK_SIZE_N]，每个元素对应B矩阵的一个位置）
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )   # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    b_zp_num = 128

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)  # 新增：K 方向迭代次数

    # 按 K 维度分块遍历，处理每个 block 的子矩阵
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not even_Ks:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        # 加载A矩阵的当前K子块（只加载有效token和有效K区域）
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # 加载B矩阵的当前K子块（专家权重，int8类型）
        b_raw = tl.load(b_ptrs)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]
        # 计算当前 block 的量化 scale 指针，按 group_size 分组
        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )   # [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # 取出 scale 并转为 float32
        b_scale = b_scale.to(tl.float32)

        # === （3.简化kernel 逻辑）仅测int8的量化
        # 对 int8 权重做反量化：(b - 128) * scale
        b_deq = ((b_raw.to(tl.float32) - b_zp_num) * b_scale).to(compute_type) # [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # --------------------------
        # 4. 存储当前 K 子块的调试数据
        # --------------------------
        # 4.1 dbg_a：按 (pid_m, k) 索引存储 A 子块（[BLOCK_SIZE_M, BLOCK_SIZE_K]）
        dbg_a_offset = (pid_m * num_k_iter + k) * BLOCK_SIZE_M * BLOCK_SIZE_K + \
                       tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        tl.store(dbg_a_ptr + dbg_a_offset, a, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K))

        # 计算B调试数据的存储偏移（核心修复：基于全局K位置）
        expert_stride = K * N  # 每个专家的存储步长（K*N）
        # 全局K维度偏移对应的存储位置 = 全局K偏移 * N（每个K元素跨N个位置）
        global_k_storage_offset = k * BLOCK_SIZE_K * N
        # 当前N块的偏移 = pid_n * BLOCK_SIZE_N
        n_block_offset = pid_n * BLOCK_SIZE_N
        # 块内K和N的偏移
        inner_offset = tl.arange(0, BLOCK_SIZE_K)[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :]

        # 最终调试数据存储位置 = 专家偏移 + 全局K偏移 + N块偏移 + 块内偏移
        dbg_b_raw_offset = (
            off_experts * expert_stride
            + global_k_storage_offset
            + n_block_offset
            + inner_offset
        )

        # 存储B相关调试数据
        tl.store(dbg_b_raw_ptr + dbg_b_raw_offset, b_raw.to(tl.float32), mask=k_mask)
        tl.store(dbg_scale_ptr + dbg_b_raw_offset, b_scale, mask=k_mask)
        tl.store(dbg_b_deq_ptr + dbg_b_raw_offset, b_deq, mask=k_mask)

        accumulator = tl.dot(a, b_deq, acc=accumulator)

        # 指针前移，准备下一个 K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    
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
    # device = "cuda"

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

    # ---- 调试输出缓冲区 ----
    # --------------------------
    # 2. 初始化调试缓冲区（匹配 kernel 存储逻辑）
    # --------------------------
    num_pid_m = triton.cdiv(EM, BLOCK_SIZE_M)  # 32 / 16 = 2（M 维度分块数）
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)  # 32 / 16 = 2（N 维度分块数）
    num_k_iter = triton.cdiv(K, BLOCK_SIZE_K)  # 64 / 16 = 4（K 维度迭代次数）

    # dbg_offs_token：存储 kernel 加载的 token 索引（与 sorted_token_ids 长度一致）
    dbg_offs_token = torch.empty(EM, dtype=torch.int64, device=device)

    # dbg_a：存储 kernel 加载的 A 数据（shape: [num_pid_m, num_k_iter, BLOCK_SIZE_M, BLOCK_SIZE_K]）
    dbg_a = torch.zeros((num_pid_m, num_k_iter, BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=torch.float32, device=device)
    num_experts = B.shape[0]  # 2（专家数量）

    # dbg_b_raw/dbg_scale/dbg_b_deq：存储 B 相关数据（shape: [num_experts, K, N]）
    dbg_b_raw = torch.zeros((num_experts, K, N), dtype=torch.float32, device=device)
    dbg_scale = torch.zeros((num_experts, K, N), dtype=torch.float32, device=device)
    dbg_b_deq = torch.zeros((num_experts, K, N), dtype=torch.float32, device=device)

    # 新增：accumulator中间值缓冲区（[num_pid_m, num_k_iter, BLOCK_SIZE_M, BLOCK_SIZE_N]）
    dbg_accumulator = torch.zeros((num_pid_m, num_k_iter, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)

    test_store_result_kernel[grid](
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
        dbg_a_ptr=dbg_a,
        dbg_b_raw_ptr=dbg_b_raw,
        dbg_scale_ptr=dbg_scale,
        dbg_b_deq_ptr=dbg_b_deq,
        dbg_offs_token_ptr=dbg_offs_token,
        dbg_accumulator_ptr=dbg_accumulator,
    )

    print("=" * 50)
    print(f"Before cast dtypes: A={A.dtype}, B={B.dtype}, B_scale={B_scale.dtype}, C={C.dtype}")
    # A=torch.float32, B=torch.int8, B_scale=torch.float32, C=torch.float32
    print("NPU计算结果：")
    print(f"C矩阵形状: {C.shape}, dtype={C.dtype}")

    token_idx = 0
    expert_ids = 0

    # 获取输入向量和权重矩阵
    a_vector = A[token_idx].float()
    b_matrix = B[expert_ids].float()
    scale_matrix = B_scale[expert_ids]

    # 手动反量化
    group_size = 16
    b_dequantized = torch.zeros_like(b_matrix)
    for n in range(b_matrix.shape[0]):  # 遍历行 32
        for k in range(b_matrix.shape[1]): # 遍历列 64
            group_idx = k // group_size
            b_dequantized[n, k] = (b_matrix[n, k] - 128) * scale_matrix[n, group_idx]
    # 手动计算矩阵乘法
    manual_result = torch.matmul(a_vector, b_dequantized.T)

    print("手动计算结果：", manual_result)
    print("Kernel计算结果：", C[token_idx, 0])
    print("差异", torch.abs(manual_result - C[token_idx, 0].float()).max())
    
    # --------------------------
    # 4. 验证 dbg 与原变量的一致性
    # --------------------------
    print("=" * 80)
    print("1. 基础参数验证")
    print(f"   - 原 sorted_token_ids: {sorted_token_ids.cpu().tolist()}")
    print(f"   - dbg_offs_token:     {dbg_offs_token.cpu().tolist()}")
    print(f"   - token 索引一致性: {torch.equal(sorted_token_ids, dbg_offs_token)}")
    print()

    # 4.1 验证 dbg_a 与原 A 的一致性
    # 原 A 按 sorted_token_ids 索引提取（kernel 加载的 A 数据）
    A_kernel_gt = A[sorted_token_ids // top_k]  # sorted_token_ids//top_k 对应原 A 的行索引
    # dbg_a 重组为 [EM, K]（与 A_kernel_gt 维度一致）
    dbg_a_reshaped = dbg_a.permute(0, 2, 1, 3).reshape(EM, K)  # [num_pid_m, BLOCK_SIZE_M, num_k_iter, BLOCK_SIZE_K] → [32,64]
    # 计算误差
    a_abs_err = torch.abs(A_kernel_gt - dbg_a_reshaped).max().item()
    a_rel_err = (torch.abs(A_kernel_gt - dbg_a_reshaped) / (torch.abs(A_kernel_gt) + 1e-8)).max().item()

    print("2. A 数据一致性验证")
    print(f"   - 原 A 按 token 索引提取后 shape: {A_kernel_gt.shape}")
    print(f"   - dbg_a 重组后 shape: {dbg_a_reshaped.shape}")
    print(f"   - 最大绝对误差: {a_abs_err:.6f}")
    print(f"   - 最大相对误差: {a_rel_err:.6f}")
    print(f"   - A 数据一致性: {a_abs_err < 1e-6}")
    print()

    # B原始数据验证（修复后）
    B_gt = B.permute(0, 2, 1).contiguous()  # [2, 32, 64] → [2, 64, 32] (专家数, K, N)
    b_raw_abs_err = torch.abs(B_gt.float() - dbg_b_raw).max().item()
    b_raw_rel_err = (torch.abs(B_gt.float() - dbg_b_raw) / (torch.abs(B_gt.float()) + 1e-8)).max().item()
    print("3. B原始数据一致性验证")
    print(f"   - 原B转置后shape: {B_gt.shape}, dbg_b_raw shape: {dbg_b_raw.shape}")
    print(f"   - 最大绝对误差: {b_raw_abs_err:.6f}, 最大相对误差: {b_raw_rel_err:.6f}")
    print(f"   - 一致性: {b_raw_abs_err < 1e-6}")
    print("   - 前2x2数据对比:")
    print(f"     原B: \n{B_gt[0, :2, :2].cpu().numpy()}")
    print(f"     dbg: \n{dbg_b_raw[0, :2, :2].cpu().numpy()}")
    print()

    # B_scale验证（修复后）
    B_scale_expand = B_scale.unsqueeze(-1).repeat(1, 1, 1, group_size)  # [2, 32, 4, 16]
    B_scale_gt = B_scale_expand.reshape(num_experts, N, K).permute(0, 2, 1).contiguous()  # [2, 64, 32]
    scale_abs_err = torch.abs(B_scale_gt - dbg_scale).max().item()
    print("4. B_scale数据一致性验证")
    print(f"   - 扩展转置后shape: {B_scale_gt.shape}, dbg_scale shape: {dbg_scale.shape}")
    print(f"   - 最大绝对误差: {scale_abs_err:.6f}, 一致性: {scale_abs_err < 1e-6}")
    print()

    # B反量化验证（修复后）
    B_deq_gt = ((B_gt.float() - 128) * B_scale_gt).contiguous()
    b_deq_abs_err = torch.abs(B_deq_gt - dbg_b_deq).max().item()
    print("5. B反量化数据一致性验证")
    print(f"   - 手动计算shape: {B_deq_gt.shape}, dbg_b_deq shape: {dbg_b_deq.shape}")
    print(f"   - 最大绝对误差: {b_deq_abs_err:.6f}, 一致性: {b_deq_abs_err < 1e-6}")
    print("=" * 80)


    torch.save({
        "A": A,
        "B": B,
        "B_scale": B_scale,
        "dbg_a": dbg_a,
        "dbg_b_raw": dbg_b_raw,
        "dbg_scale": dbg_scale,
        "dbg_b_deq": dbg_b_deq,
        "dbg_offs_token": dbg_offs_token,
        "dbg_accumulator": dbg_accumulator,
    }, "data_dbg_cpu.pt")


def compare_results():
    cpu_data = torch.load("data_dbg_cpu.pt", map_location="cpu", weights_only=True)
    cpu_dbg_accumulator = cpu_data["dbg_accumulator"]

    cuda_data = torch.load("data_dbg_cuda.pt", map_location="cpu", weights_only=True)
    cuda_dbg_accumulator = cuda_data["dbg_accumulator"]

    torch.testing.assert_close(cpu_dbg_accumulator, cuda_dbg_accumulator, rtol=1e-2, atol=1e-2)
    print("CPU和NPU的dbg_accumulator结果一致！")




if __name__ == "__main__":
    test_fn()
    # compare_results()
