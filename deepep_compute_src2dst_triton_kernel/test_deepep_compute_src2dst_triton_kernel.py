import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

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


def save_inputs_outputs(path: str, num_toks: int = 8, BLOCK_SIZE: int = 4):
    # 创建输入张量
    reorder_ids = torch.arange(num_toks, dtype=torch.int32, device="npu")
    src2dst = torch.zeros(num_toks, dtype=torch.int32, device="npu")
    
    # 动态计算无效 token 的数量
    num_invalid_tokens = torch.sum(reorder_ids == -1)  # 假设 -1 表示无效 token
    num_minus_one = torch.tensor([num_invalid_tokens], dtype=torch.int32, device="npu")

    # 调用 Triton 内核
    src2dst = deepep_compute_src2dst_impl(
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_minus_one=num_minus_one,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    print(">> reorder_ids:", reorder_ids.cpu().numpy())
    print(">> Compute src2dst:", src2dst.cpu().numpy())

    # 保存输入输出
    torch.save({
        "reorder_ids": reorder_ids.cpu(),
        "src2dst": src2dst.cpu(),
        "num_minus_one": num_minus_one.cpu(),
        "BLOCK_SIZE": BLOCK_SIZE
    }, path)


def run_and_compare(path):
    data = torch.load(path)
    reorder_ids = data["reorder_ids"].to("npu")
    src2dst = torch.zeros_like(data["src2dst"]).to("npu")
    num_minus_one = data["num_minus_one"].to("npu")
    BLOCK_SIZE = data["BLOCK_SIZE"]

    # 重新计算输出
    src2dst = deepep_compute_src2dst_impl(
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_minus_one=num_minus_one,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 检查结果
    expected_output = data["src2dst"].to("npu")
    check_accuracy(src2dst, expected_output)


if __name__ == "__main__":
    # 编译测试
    path = "compute_src2dst_npu_output.pt"
    save_inputs_outputs(path)

    # 对比cuda和triton-ascend的输出
    # path = "compute_src2dst_cuda_output.pt"
    # run_and_compare(path)
