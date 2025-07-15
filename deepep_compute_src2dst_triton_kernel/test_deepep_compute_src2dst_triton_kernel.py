import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def deepep_compute_src2dst_triton_kernel(
    reorder_ids_ptr, src2dst_ptr, num_toks, num_minus_one, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids_ptr + dst_id, mask=mask)
    num_invalid = tl.load(num_minus_one)
    tl.store(src2dst_ptr + src_id, dst_id - num_invalid, mask=mask)

def deepep_compute_src2dst_impl(
    reorder_ids: torch.Tensor,  # (num_toks,)
    src2dst: torch.Tensor,      # (num_toks,)
    num_minus_one: torch.Tensor,  # Scalar (1,)
    BLOCK_SIZE: int = 512,
):
    """
    Compute the mapping from source indices to destination indices.

    Args:
        reorder_ids: Tensor containing the reordered source indices (num_toks,).
        src2dst: Output tensor for the computed mapping (num_toks,).
        num_minus_one: Scalar tensor containing the number of invalid tokens (1,).
        BLOCK_SIZE: Block size for Triton kernel.
    """
    num_toks = reorder_ids.shape[0]

    assert reorder_ids.device == src2dst.device
    assert reorder_ids.dtype == torch.int32
    assert src2dst.dtype == torch.int32
    assert num_minus_one.numel() == 1

    grid = lambda meta: (triton.cdiv(num_toks, BLOCK_SIZE),)

    # Launch the Triton kernel
    deepep_compute_src2dst_triton_kernel[grid](
        reorder_ids_ptr=reorder_ids,
        src2dst_ptr=src2dst,
        num_toks=num_toks,
        num_minus_one=num_minus_one,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return src2dst

# 初始化输入张量
num_toks = 8
BLOCK_SIZE = 4

# Create input tensors
reorder_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device="npu")
src2dst = torch.zeros(num_toks, dtype=torch.int32, device="npu")
num_minus_one = torch.tensor([2], dtype=torch.int32, device="npu")  # Number of invalid tokens

# Call the function
src2dst = deepep_compute_src2dst_impl(
    reorder_ids=reorder_ids,
    src2dst=src2dst,
    num_minus_one=num_minus_one,
    BLOCK_SIZE=BLOCK_SIZE,
)

# Print results
print("Reorder IDs:")
print(reorder_ids)
print("Computed Src2Dst Mapping:")
print(src2dst)