import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

@triton.jit
def deepep_permute_triton_kernel(
    input_ptr,         # input tensor (src_len, hidden_size)
    gateup_input_ptr,  # output tensor (dst_len, hidden_size)
    src2dst_ptr,       # mapping from source to destination indices (src_len, topk)
    topk_ids_ptr,      # top-k expert ids (src_len, topk)
    a1_scales_ptr,     # optional scaling factors (if needed)
    topk: tl.constexpr, 
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    # Get the current source index
    src_idx = tl.program_id(0)

    # Compute pointers for src2dst and topk_ids
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    # Compute pointer for the source data
    src_ptr = input_ptr + src_idx * hidden_size

    # Process the hidden_size dimension in blocks
    for start_offset in range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        # Load input data for the current block
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)

        # Iterate over the top-k experts
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)  # Load destination index
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


def deepep_permute_impl(
    input: torch.Tensor,          # (src_len, hidden_size)
    gateup_input: torch.Tensor,   # (dst_len, hidden_size)
    src2dst: torch.Tensor,        # (src_len, topk)
    topk_ids: torch.Tensor,       # (src_len, topk)
    a1_scales: torch.Tensor,      # Optional (src_len,)
    topk: int,
    hidden_size: int,
    BLOCK_SIZE: int = 512,
):
    """
    Perform permutation of input data based on src2dst mapping.

    Args:
        input: Input tensor (src_len, hidden_size).
        gateup_input: Output tensor (dst_len, hidden_size).
        src2dst: Mapping from source to destination indices (src_len, topk).
        topk_ids: Top-k expert ids (src_len, topk).
        a1_scales: Optional scaling factors (src_len,).
        topk: Number of top-k experts.
        hidden_size: Hidden size dimension.
        BLOCK_SIZE: Block size for Triton kernel.
    """
    assert input.shape[1] == hidden_size
    assert gateup_input.shape[1] == hidden_size
    assert src2dst.shape[1] == topk
    assert topk_ids.shape[1] == topk

    grid = lambda meta: (input.shape[0],)

    # Launch the Triton kernel
    deepep_permute_triton_kernel[grid](
        input_ptr=input,
        gateup_input_ptr=gateup_input,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        a1_scales_ptr=a1_scales,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return gateup_input

def save_inputs_outputs(path):
    torch.manual_seed(42)
    src_len, dst_len, hidden_size, topk = 8, 16, 128, 2
    BLOCK_SIZE = 64
    input = torch.randn(src_len, hidden_size, device="npu", dtype=torch.float32)
    gateup_input = torch.zeros(dst_len, hidden_size, device="npu", dtype=torch.float32)
    src2dst = torch.randperm(dst_len, device="npu")[:src_len * topk].reshape(src_len, topk)
    topk_ids = torch.randint(0, 10, (src_len, topk), device="npu", dtype=torch.int32)
    a1_scales = torch.rand(src_len, device="npu", dtype=torch.float32)

    output = deepep_permute_impl(
        input=input,
        gateup_input=gateup_input,
        src2dst=src2dst,
        topk_ids=topk_ids,
        a1_scales=a1_scales,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch.save({
        "input": input.cpu(),
        "src2dst": src2dst.cpu(),
        "topk_ids": topk_ids.cpu(),
        "a1_scales": a1_scales.cpu(),
        "output": output.cpu(),
    }, path)


def run_and_compare(path, BLOCK_SIZE: int = 64):
    data = torch.load(path)
    input = data["input"].to("npu")
    src2dst = data["src2dst"].to("npu")
    topk_ids = data["topk_ids"].to("npu")
    a1_scales = data["a1_scales"].to("npu")
    expected_output = data["output"].to("npu")

    gateup_input = torch.zeros_like(expected_output)
    output = deepep_permute_impl(input, gateup_input, src2dst, topk_ids, a1_scales, topk_ids.shape[1], input.shape[1], BLOCK_SIZE)

    check_accuracy(output, expected_output)

if __name__ == "__main__":
    path = "deepep_permute_float_cuda_output.pt"
    run_and_compare(path)       # 对比cuda和triton-ascend的输出
    # >>> Compare Type: float16
    # 精度达标 (0/2048, 0.000000% <= 0.100000%)
    # >>> Compare Type: float32
    # 精度达标 (0/2048, 0.000000% <= 0.010000%)
