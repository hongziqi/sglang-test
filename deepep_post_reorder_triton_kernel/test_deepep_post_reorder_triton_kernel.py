import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def deepep_post_reorder_triton_kernel(
    down_output_ptr,      # input tensor (dst_len, hidden_size)
    output_ptr,           # output tensor (src_len, hidden_size)
    src2dst_ptr,          # mapping from source to destination indices (src_len, topk)
    topk_ids_ptr,         # top-k expert ids (src_len, topk)
    topk_weights_ptr,     # top-k weights (src_len, topk)
    topk: tl.constexpr, 
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty

    # Get the current source index
    src_idx = tl.program_id(0)

    # Compute pointers for src2dst, topk_ids, and topk_weights
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    # Compute pointer for the output data
    store_ptr = output_ptr + src_idx * hidden_size

    # Process the hidden_size dimension in blocks
    for start_offset in range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        # Initialize sum vector for accumulation
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)

        # Iterate over the top-k experts
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)  # Load destination index
            if dst_idx >= 0:
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)  # Load weight
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)  # Load input data
                sum_vec += in_data * weigh_scale  # Weighted sum

        # Store the accumulated result
        tl.store(store_ptr + offset, sum_vec, mask=mask)


def deepep_post_reorder_impl(
    down_output: torch.Tensor,   # (dst_len, hidden_size)
    output: torch.Tensor,        # (src_len, hidden_size)
    src2dst: torch.Tensor,       # (src_len, topk)
    topk_ids: torch.Tensor,      # (src_len, topk)
    topk_weights: torch.Tensor,  # (src_len, topk)
    topk: int,
    hidden_size: int,
    BLOCK_SIZE: int = 512,
):
    """
    Perform post-reordering of down-projected outputs based on src2dst mapping.

    Args:
        down_output: Down-projected output tensor (dst_len, hidden_size).
        output: Final output tensor (src_len, hidden_size).
        src2dst: Mapping from source to destination indices (src_len, topk).
        topk_ids: Top-k expert ids (src_len, topk).
        topk_weights: Top-k weights (src_len, topk).
        topk: Number of top-k experts.
        hidden_size: Hidden size dimension.
        BLOCK_SIZE: Block size for Triton kernel.
    """
    assert down_output.shape[1] == hidden_size
    assert output.shape[1] == hidden_size
    assert src2dst.shape[1] == topk
    assert topk_ids.shape[1] == topk
    assert topk_weights.shape[1] == topk

    grid = lambda meta: (output.shape[0],)

    # Launch the Triton kernel
    deepep_post_reorder_triton_kernel[grid](
        down_output_ptr=down_output,
        output_ptr=output,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        topk_weights_ptr=topk_weights,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def save_inputs_outputs(path, src_len: int = 8, dst_len: int = 16, hidden_size: int = 128, topk: int = 2, BLOCK_SIZE: int = 64):
    torch.manual_seed(42)

    down_output = torch.randn(dst_len, hidden_size, device="npu", dtype=torch.float16)
    output = torch.zeros(src_len, hidden_size, device="npu", dtype=torch.float16)
    src2dst = torch.randint(0, dst_len, (src_len, topk), device="npu", dtype=torch.int32)
    topk_ids = torch.randint(0, 10, (src_len, topk), device="npu", dtype=torch.int32)
    topk_weights = torch.rand(src_len, topk, device="npu", dtype=torch.float16)

    # 调用函数
    output = deepep_post_reorder_impl(
        down_output=down_output,
        output=output,
        src2dst=src2dst,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.save({
        "down_output": down_output.cpu(),
        "src2dst": src2dst.cpu(),
        "topk_ids": topk_ids.cpu(),
        "topk_weights": topk_weights.cpu(),
        "output": output.cpu(),
    }, path)


def check_accuracy(output: torch.Tensor, expected: torch.Tensor):
    # 根据 dtype 自动判定阈值
    dtype = expected.dtype
    if dtype == torch.float16:
        print(">>> Compare Type: float16")
        rtol, atol, max_fail_ratio = 1e-3, 1e-3, 1e-3  # 双千分之一
    elif dtype == torch.float32:
        print(">>> Compare Type: float32")
        rtol, atol, max_fail_ratio = 1e-4, 1e-4, 1e-4  # 双万分之一
    elif dtype in [torch.int8, torch.uint8]:
        print(">>> Compare Type: int8 | uint8")
        rtol, atol, max_fail_ratio = 1e-3, 1, 1e-3     # 容差为1
    else:
        raise ValueError(f"Unsupported dtype for accuracy check: {dtype}")

    # 计算误差
    abs_diff = (output - expected).abs()
    rel_diff = abs_diff / (expected.abs() + 1e-6)
    fail_mask = (abs_diff > atol) & (rel_diff > rtol)

    total = output.numel()
    fail = fail_mask.sum().item()
    fail_ratio = fail / total

    # 打印最大误差点
    max_abs = abs_diff.max().item()
    if max_abs > 0:
        max_idx_flat = torch.argmax(abs_diff).item()
        i, j = divmod(max_idx_flat, output.shape[1])
        print(f"Max diff at [{i}, {j}]: test={output[i, j].item()}, "
            f"ref={expected[i, j].item()}, "
            f"abs={abs_diff[i, j].item()}, rel={rel_diff[i, j].item()}")

    # 判断是否精度达标
    if fail_ratio <= max_fail_ratio:
        print(f"精度达标 ({fail}/{total}, {fail_ratio:.6%} <= {max_fail_ratio:.6%})")
    else:
        print(f"精度不达标 ({fail}/{total}, {fail_ratio:.6%} > {max_fail_ratio:.6%})")
        idx_list = torch.nonzero(fail_mask)[:10]
        for i, j in idx_list.tolist():
            print(f"[{i},{j}]: test={output[i, j].item():.6f}, "
                  f"ref={expected[i, j].item():.6f}, "
                  f"diff={abs_diff[i, j].item():.6f}, rel={rel_diff[i, j].item():.6f}")

    return fail_ratio


def run_and_compare(path, BLOCK_SIZE: int = 64):
    data = torch.load(path)
    down_output = data["down_output"].to("npu")
    src2dst = data["src2dst"].to("npu")
    topk_ids = data["topk_ids"].to("npu")
    topk_weights = data["topk_weights"].to("npu")
    expected_output = data["output"].to("npu")

    output = deepep_post_reorder_impl(
        down_output=down_output,
        output=torch.zeros_like(expected_output),
        src2dst=src2dst,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk=topk_ids.shape[1],
        hidden_size=down_output.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    check_accuracy(output, expected_output)
    

if __name__ == "__main__":
    path = "deepep_post_reorder_float_cuda_output.pt"
    run_and_compare(path)       # 对比cuda和triton-ascend的输出
    # >>> Compare Type: float16
    # Max diff at [2, 51]: test=-2.0, ref=-2.001953125, abs=0.001953125, rel=0.0009756088256835938
    # 精度达标 (0/1024, 0.000000% <= 0.100000%)
    # >>> Compare Type: float32
    # Max diff at [3, 90]: test=2.2100167274475098, ref=2.2100164890289307, abs=2.384185791015625e-07, rel=1.0788085802460046e-07
    # 精度达标 (0/1024, 0.000000% <= 0.010000%)