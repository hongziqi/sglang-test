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
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
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
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not even_Ks:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def test_and_save_gpu_result():
    test_data = torch.load("62_simple_fused_moe_test_data.pt", map_location="cpu", weights_only=True)

    device = "npu:0"

    A = test_data["A"].to(device)
    B = test_data["B"].to(device)
    C = test_data["C"].to(device)
    B_scale = test_data["B_scale"].to(device)
    B_zp = test_data["B_zp"].to(device) if test_data["B_zp"] is not None else None
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
    BLOCK_SIZE_M = test_data["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = test_data["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = test_data["BLOCK_SIZE_K"]
    GROUP_SIZE_M = test_data["GROUP_SIZE_M"]

    # 打印所有输入参数信息
    print("=" * 50)
    print("输入参数信息：")
    print("=" * 50)
    print(f"A.shape={A.shape}, dtype={A.dtype}, device={A.device}, value={A}")
    print(f"B.shape={B.shape}, dtype={B.dtype}, device={B.device}, value={B}")

    # 检查B张量是否真的全零
    print(f"B张量中非零元素的数量：{torch.count_nonzero(B).item()}")
    print("B张量的形状", B.shape)
    print("B张量的最小值", torch.min(B).item())
    print("B张量的最大值", torch.max(B).item())

    print(f"has_zp={has_zp}") # False
    print(f"use_int4_w4a16={use_int4_w4a16}") # False
    print(f"use_int8_w8a16={use_int8_w8a16}") # True


    
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) *
        triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    C_initial = C.clone()

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
    )

    print("=" * 50)
    print("NPU计算结果：")
    print(f"C矩阵形状: {C.shape}, dtype={C.dtype}")
    print(f"C矩阵最小值: {torch.min(C).item()}")
    print(f"C矩阵最大值: {torch.max(C).item()}")
    print(f"C矩阵平均值: {torch.mean(C).item()}")
    print(f"C矩阵中非零元素比例: {torch.count_nonzero(C).item() / C.numel():.6f}")
    print(f"C矩阵的值：{C.cpu().tolist()}")

    token_idx = 0
    expert_ids = 0

    # 获取输入向量和权重矩阵
    a_vector = A[token_idx].float()
    b_matrix = B[expert_ids].float()
    scale_matrix = B_scale[expert_ids]

    # 手动反量化
    group_size = 16
    b_dequantized = torch.zeros_like(b_matrix)
    for n in range(b_matrix.shape[0]):
        for k in range(b_matrix.shape[1]):
            group_idx = k // group_size
            b_dequantized[n, k] = (b_matrix[n, k] - 128) * scale_matrix[n, group_idx]
    
    # 手动计算矩阵乘法
    manual_result = torch.matmul(a_vector, b_dequantized.T)

    print("手动计算结果：", manual_result)
    print("Kernel计算结果：", C[token_idx, 0])
    print("差异", torch.abs(manual_result - C[token_idx, 0].float()).max())
    # 手动计算结果： tensor([ -919.4400, -1775.6799,  -856.2401, -1649.2799,  -793.0400, -1522.8800,
    #      -729.8400, -1396.4801,  -666.6400, -1270.0800,  -603.4400, -1143.6799,
    #      -540.2400, -1081.7920,  -662.8960, -1544.1920,  -868.4960, -1846.5920,
    #      -968.4960, -2008.1921, -1014.9921, -2030.7841, -1009.3920, -1983.2321,
    #      -966.4160, -1876.0320,  -906.8000, -1750.4001,  -843.6000, -1624.0000,
    #      -780.4000, -1497.6000], device='npu:0')
    # Kernel计算结果： tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #         0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
    # 差异 tensor(2030.7841, device='npu:0')


if __name__ == "__main__":
    test_and_save_gpu_result()