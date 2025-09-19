import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
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

    # # ---- 调试开关与目标子块 ----
    # DBG_ENABLE: tl.constexpr,         # 是否启用调试写
    # DBG_PID_M: tl.constexpr,          # 望检查的 pid_m
    # DBG_PID_N: tl.constexpr,          # 望检查的 pid_n
    # DBG_K: tl.constexpr,              # 望检查的第几个 K block（0 表示第一个）
    # # ---- 调试输出缓冲区（全部视为 row-major 连续）----
    # dbg_a_ptr,        # float32 [BLOCK_SIZE_M, BLOCK_SIZE_K]
    # dbg_b_raw_ptr,    # float32 [BLOCK_SIZE_K, BLOCK_SIZE_N]（把原始 int8 转成 float 写）
    # dbg_scale_ptr,    # float32 [BLOCK_SIZE_K, BLOCK_SIZE_N]
    # dbg_b_deq_ptr,    # float32 [BLOCK_SIZE_K, BLOCK_SIZE_N]
    # dbg_acc_ptr,      # float32 [BLOCK_SIZE_M, BLOCK_SIZE_N]

):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
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
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    # 生成token有效mask（仅处理小于num_valid_tokens的真实token）
    token_mask = offs_token < num_valid_tokens

    # 通过 expert_ids_ptr 得到当前 block 的专家 id（off_experts）
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # N维度的块内偏移（当前块在N维度的起始位置）
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    # K维度的块内偏移（每次循环处理的K子块）
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 输入token的指针（形状：[BLOCK_SIZE_M, BLOCK_SIZE_K]，每个元素对应A矩阵的一个位置）
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    
    # === （1.简化kernel 逻辑）仅测int8的量化
    # 专家权重的指针（形状：[BLOCK_SIZE_K, BLOCK_SIZE_N]，每个元素对应B矩阵的一个位置）
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    b_zp_num = 128
    # === （1.简化kernel 逻辑）仅测int8的量化
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # 初始化累加器 accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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
        )
        # 加载B矩阵的当前K子块（专家权重，int8类型）
        b_raw = tl.load(b_ptrs)
        # 计算当前 block 的量化 scale 指针，按 group_size 分组
        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)

        # 取出 scale 并转为 float32
        b_scale = b_scale.to(tl.float32)

        # === （3.简化kernel 逻辑）仅测int8的量化
        # 对 int8 权重做反量化：(b - 128) * scale
        b_deq = ((b_raw.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)

        # ---- 调试写：仅在目标子块/目标 k 次迭代写一次 ----
        # if DBG_ENABLE and (k == DBG_K):  # 这两个是 tl.constexpr，可做编译期裁剪
        #     cond = (pid_m == DBG_PID_M) & (pid_n == DBG_PID_N)  # 运行期布尔，用按位与
        #     # a tile -> [M,K]
        #     ra = tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        #     tl.store(dbg_a_ptr + ra, a.to(tl.float32), mask=cond)

        #     # b_raw/b_scale/b_deq tile -> [K,N]（行主序）
        #     rb = tl.arange(0, BLOCK_SIZE_K)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
        #     tl.store(dbg_b_raw_ptr + rb, b_raw.to(tl.float32), mask=cond)
        #     tl.store(dbg_scale_ptr + rb, b_scale, mask=cond)
        #     tl.store(dbg_b_deq_ptr + rb, b_deq.to(tl.float32), mask=cond)

        # ---- 调试写结束 ----

        # 矩阵乘法并累加（A[M,K] * B[K,N] -> 累加至accumulator[M,N]）
        accumulator = tl.dot(a, b_deq, acc=accumulator)

        # 指针前移，准备下一个 K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        # === （3.简化kernel 逻辑）仅测int8的量化

    # 将累加结果转为目标类型（如 fp16）
    accumulator = accumulator.to(compute_type)
    # 输出N维度的偏移
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 输出矩阵的指针（形状：[BLOCK_SIZE_M, BLOCK_SIZE_N]）
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    # 输出有效mask（只写入有效token和有效N区域）
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    # 存储结果
    tl.store(c_ptrs, accumulator, mask=c_mask)



def test_and_save_gpu_result():
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


    # ---- 调试开关与目标子块 ----
     # 选择要检查的子块：第 0 个 pid_m、pid_n，和第 0 个 K-block 
    DBG_ENABLE = True
    DBG_PID_M = 0
    DBG_PID_N = 0
    DBG_K = 0

    dbg_a = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=torch.float32, device=device)
    dbg_b_raw = torch.empty((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    dbg_scale = torch.empty((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    dbg_b_deq = torch.empty((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    dbg_acc = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device=device)
    # ---- 调试输出缓冲区 ----

    fused_moe_kernel_gptq_awq[grid](
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

    torch.npu.synchronize()
    # ---------
    # print("-" * 50)
    # print("调试输出：")
    # print(f"dbg_a (A tile) shape: {dbg_a.shape}, dtype={dbg_a.dtype}")
    # print(f"dbg_b_raw (B raw tile) shape: {dbg_b_raw.shape}, dtype={dbg_b_raw.dtype}")
    # print(f"dbg_scale (scale tile) shape: {dbg_scale.shape}, dtype={dbg_scale.dtype}")
    # print(f"dbg_b_deq (B deq tile) shape: {dbg_b_deq.shape}, dtype={dbg_b_deq.dtype}")
    # print(f"dbg_acc (accumulator tile) shape: {dbg_acc.shape}, dtype={dbg_acc.dtype}")
    # print("dbg_a:", dbg_a)
    # print("dbg_b_raw:", dbg_b_raw)
    # print("dbg_scale:", dbg_scale)
    # print("dbg_b_deq:", dbg_b_deq)
    # print("dbg_acc:", dbg_acc)
    # print("-" * 50)
     # ===== 在 PyTorch 端构造同位置的“期望”tile 并比对 =====
    # 当前块 token 索引与 mask
    offs_token_id = DBG_PID_M * BLOCK_SIZE_M + torch.arange(BLOCK_SIZE_M, device=device, dtype=torch.long)
    offs_token = sorted_token_ids[offs_token_id]
    token_mask = offs_token < num_valid_tokens
    base_k = DBG_K * BLOCK_SIZE_K
    n0 = DBG_PID_N * BLOCK_SIZE_N
    n1 = min(n0 + BLOCK_SIZE_N, N)

    # 期望的 A 子块（注意 A 的行索引使用 offs_token//top_k）
    a_ref = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=torch.float32, device=device)
    valid_rows = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
    if valid_rows.numel() > 0:
        a_rows = (offs_token[valid_rows] // top_k).to(torch.long)
        a_ref[valid_rows, :n1-n0] = A.index_select(0, a_rows)[:, base_k:base_k+BLOCK_SIZE_K]

    # 期望的 B 原始子块（布局与 kernel 一致：B[e, n, k]）
    e = expert_ids[DBG_PID_M].item()
    b_raw_ref = B[e, n0:n1, base_k:base_k+BLOCK_SIZE_K].transpose(0, 1).contiguous().to(torch.float32)  # [K,N]

    # 期望的 scale 子块（按 group_size 聚合 K 维）
    scale_ref = B_scale[e, n0:n1, (base_k // group_size):((base_k+BLOCK_SIZE_K+group_size-1)//group_size)]
    # 展开到 [K,N]
    k_idx = torch.arange(base_k, base_k+BLOCK_SIZE_K, device=device) // group_size
    scale_ref = B_scale[e, n0:n1, k_idx - (base_k // group_size)].transpose(0, 1).contiguous().to(torch.float32)

    b_deq_ref = (b_raw_ref - 128.0) * scale_ref

    print("check a tile max|diff|:", (dbg_a - a_ref).abs().max().item())
    print("check b_raw tile max|diff|:", (dbg_b_raw - b_raw_ref).abs().max().item())
    print("check scale tile max|diff|:", (dbg_scale - scale_ref).abs().max().item())
    print("check b_deq tile max|diff|:", (dbg_b_deq - b_deq_ref).abs().max().item())

    # ---------


    print("=" * 50)
    print(f"Before cast dtypes: A={A.dtype}, B={B.dtype}, B_scale={B_scale.dtype}, C={C.dtype}")
    # A=torch.float32, B=torch.int8, B_scale=torch.float32, C=torch.float32
    print("NPU计算结果：")
    print(f"C矩阵形状: {C.shape}, dtype={C.dtype}")
    print(f"C矩阵中非零元素比例: {torch.count_nonzero(C).item() / C.numel():.6f}")
    # print(f"C矩阵的值：{C.cpu().tolist()}")

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
    # print("a_vector:", a_vector)
    # a_vector: tensor([0.0000, 0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.0600, 0.0700, 0.0800,
    #     0.0900, 0.1000, 0.1100, 0.1200, 0.1300, 0.1400, 0.1500, 0.1600, 0.1700,
    #     0.1800, 0.1900, 0.2000, 0.2100, 0.2200, 0.2300, 0.2400, 0.2500, 0.2600,
    #     0.2700, 0.2800, 0.2900, 0.3000, 0.3100, 0.3200, 0.3300, 0.3400, 0.3500,
    #     0.3600, 0.3700, 0.3800, 0.3900, 0.4000, 0.4100, 0.4200, 0.4300, 0.4400,
    #     0.4500, 0.4600, 0.4700, 0.4800, 0.4900, 0.5000, 0.5100, 0.5200, 0.5300,
    #     0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200,
    #     0.6300], device='npu:0')
    # print("反量化后的权重矩阵：", b_dequantized)
    # 反量化后的权重矩阵： tensor([[ -19.2000,  -19.1000,  -19.0000,  ...,  -52.4000,  -52.0000,
    #       -51.6000],
    #     [ -37.4000,  -37.2000,  -37.0000,  ..., -100.8000, -100.0000,
    #       -99.2000],
    #     [ -18.2000,  -18.1000,  -18.0000,  ...,  -48.4000,  -48.0000,
    #       -47.6000],
    #     ...,
    #     [ -35.0000,  -34.8000,  -34.6000,  ...,  -91.2000,  -90.4000,
    #       -89.6000],
    #     [ -17.0000,  -16.9000,  -16.8000,  ...,  -43.6000,  -43.2000,
    #       -42.8000],
    #     [ -33.0000,  -32.8000,  -32.6000,  ...,  -83.2000,  -82.4000,
    #       -81.6000]], device='npu:0')
    

    print("手动计算结果：", manual_result)
    print("Kernel计算结果：", C[token_idx, 0])
    print("差异", torch.abs(manual_result - C[token_idx, 0].float()).max())
    # 手动计算结果： tensor([ -919.4400, -1775.6799,  -856.2401, -1649.2799,  -793.0400, -1522.8800,
    #         -729.8400, -1396.4801,  -666.6400, -1270.0800,  -603.4400, -1143.6799,
    #         -540.2400, -1081.7920,  -662.8960, -1544.1920,  -868.4960, -1846.5920,
    #         -968.4960, -2008.1921, -1014.9921, -2030.7841, -1009.3920, -1983.2321,
    #         -966.4160, -1876.0320,  -906.8000, -1750.4001,  -843.6000, -1624.0000,
    #         -780.4000, -1497.6000], device='npu:0')
    # Kernel计算结果： tensor([3.6840e-42, 0.0000e+00, 3.7737e-42, 0.0000e+00, 3.8634e-42, 0.0000e+00,
    #         3.9531e-42, 0.0000e+00, 4.0427e-42, 0.0000e+00, 4.1324e-42, 0.0000e+00,
    #         4.2221e-42, 0.0000e+00, 4.3118e-42, 0.0000e+00, 5.1189e-42, 0.0000e+00,
    #         5.2086e-42, 0.0000e+00, 5.2983e-42, 0.0000e+00, 5.3880e-42, 0.0000e+00,
    #         5.4777e-42, 0.0000e+00, 5.5674e-42, 0.0000e+00, 5.6570e-42, 0.0000e+00,
    #         5.7467e-42, 0.0000e+00], device='npu:0')
    # 差异 tensor(2030.7841, device='npu:0')

    # CPU 模式
    # 手动计算结果： tensor([ -919.4400, -1775.6799,  -856.2401, -1649.2799,  -793.0400, -1522.8800,
    #      -729.8400, -1396.4801,  -666.6400, -1270.0800,  -603.4400, -1143.6799,
    #      -540.2400, -1081.7920,  -662.8960, -1544.1920,  -868.4960, -1846.5920,
    #      -968.4960, -2008.1921, -1014.9921, -2030.7841, -1009.3920, -1983.2321,
    #      -966.4160, -1876.0320,  -906.8000, -1750.4001,  -843.6000, -1624.0000,
    #      -780.4000, -1497.6000], device='npu:0')
    # Kernel计算结果： tensor([ -919.4399, -1775.6801,  -856.2400, -1649.2800,  -793.0400, -1522.8801,
    #         -729.8400, -1396.4800,  -666.6400, -1270.0801,  -603.4401, -1143.6801,
    #         -540.2400, -1081.7920,  -662.8960, -1544.1920,  -868.4960, -1846.5920,
    #         -968.4960, -2008.1919, -1014.9921, -2030.7841, -1009.3920, -1983.2321,
    #         -966.4160, -1876.0321,  -906.8000, -1750.3999,  -843.6000, -1624.0000,
    #         -780.4000, -1497.6001], device='npu:0')
    # 差异 tensor(0.0002, device='npu:0')


if __name__ == "__main__":
    test_and_save_gpu_result()
