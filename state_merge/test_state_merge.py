import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

def check_input(x: torch.Tensor):
    assert x.is_contiguous(), f"{str(x)} must be contiguous"


def check_dim(d, x: torch.Tensor):
    assert x.dim() == d, f"{str(x)} must be a {d}D tensor"


def check_shape(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == b.dim(), "tensors should have same dim"
    for i in range(a.dim()):
        assert a.size(i) == b.size(
            i
        ), f"tensors shape mismatch, {a.size()} and {b.size()}"


@triton.jit
def state_merge(o, m, d, other_o, other_m, other_d):
    m_max = tl.maximum(m, other_m)  # 计算最大标量
    d = d * tl.exp2(m - m_max) + other_d * tl.exp2(other_m - m_max) # 计算合并的标量
    o = o * tl.exp2(m - m_max) + other_o * tl.exp2(other_m - m_max) # 计算合并的向量
    return o, m_max, d


@triton.jit
def state_merge_kernel(
    v_a_ptr, s_a_ptr,
    v_b_ptr, s_b_ptr,
    v_out_ptr, s_out_ptr, d_out_ptr,
    num_heads, head_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs = tl.arange(0, BLOCK)
    mask = offs < head_size

    # 计算偏移
    s_offset = pid * num_heads + head_idx
    v_offset = s_offset * head_size + offs

    # 加载数据
    v_a = tl.load(v_a_ptr + v_offset, mask=mask)
    v_b = tl.load(v_b_ptr + v_offset, mask=mask)
    s_a = tl.load(s_a_ptr + s_offset)
    s_b = tl.load(s_b_ptr + s_offset)
    d = 1.0
    other_d = 1.0

    v_merged, s_merged, d_merged = state_merge(v_a, s_a, d, v_b, s_b, other_d)

    # 保存结果
    tl.store(v_out_ptr + v_offset, v_merged, mask=mask)
    tl.store(s_out_ptr + s_offset, s_merged)
    tl.store(d_out_ptr + s_offset, d_merged)


def state_merge_impl(
    v_a: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    s_a: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS]
    v_b: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    s_b: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS]
    BLOCK: int = 256,  # 每个线程块的大小
):
    """执行状态合并操作。
    Args:
        v_a: 向量 A，形状为 [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]。
        s_a: 标量 A，形状为 [NUM_TOKENS, NUM_HEADS]。
        v_b: 向量 B，形状为 [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]。
        s_b: 标量 B，形状为 [NUM_TOKENS, NUM_HEADS]。
        BLOCK: 每个线程块的大小。
    """
    check_input(v_a)
    check_input(s_a)
    check_input(v_b)
    check_input(s_b)
    check_shape(v_a, v_b)
    check_shape(s_a, s_b)

    num_tokens, num_heads, head_size = v_a.shape

    v_a_flat = v_a.contiguous().view(-1)
    v_b_flat = v_b.contiguous().view(-1)
    v_out = torch.empty_like(v_a).contiguous().view(-1)
    s_out = torch.empty_like(s_a)
    d_out = torch.empty_like(s_a)

    grid = (num_tokens, num_heads)

    state_merge_kernel[grid](
        v_a_flat, s_a,
        v_b_flat, s_b,
        v_out, s_out, d_out,
        num_heads, head_size,
        BLOCK=BLOCK
    )

    return v_out.view_as(v_a), s_out, d_out


def save_inputs_outputs(path, num_tokens: int = 1024, num_heads: int = 8, head_size: int = 64, BLOCK_SIZE: int = 256):
    # 输入规格
    # v_a (num_tokens, num_heads, head_size) float32 向量 A
    # s_a (num_tokens, num_heads) float32 标量 A
    # v_b (num_tokens, num_heads, head_size) float32 向量 B
    # s_b (num_tokens, num_heads) float32 标量 B
    v_a = torch.randn(num_tokens, num_heads, head_size, device="npu", dtype=torch.float32)
    s_a = torch.randn(num_tokens, num_heads, device="npu", dtype=torch.float32)
    v_b = torch.randn(num_tokens, num_heads, head_size, device="npu", dtype=torch.float32)
    s_b = torch.randn(num_tokens, num_heads, device="npu", dtype=torch.float32)

    v_merged, s_merged, d_merged = state_merge_impl(v_a, s_a, v_b, s_b, BLOCK=BLOCK_SIZE)

    torch.save({
        "v_merged": v_merged,
        "s_merged": s_merged,
        "d_merged": d_merged,
        "v_a": v_a,
        "s_a": s_a,
        "v_b": v_b,
        "s_b": s_b,
        "BLOCK_SIZE": BLOCK_SIZE,
    }, path)


def run_and_compare(path, rtol: float = 1e-3, atol: float = 1e-3):
    data = torch.load(path)
    v_a = data["v_a"].npu()
    s_a = data["s_a"].npu()
    v_b = data["v_b"].npu()
    s_b = data["s_b"].npu()
    BLOCK_SIZE = data["BLOCK_SIZE"]

    v_merged, s_merged, d_merged = state_merge_impl(v_a, s_a, v_b, s_b, BLOCK_SIZE)

    expected_v_merged = data["v_merged"].npu()
    expected_s_merged = data["s_merged"].npu()
    expected_d_merged = data["d_merged"].npu()

    check_accuracy(v_merged.cpu(), expected_v_merged.cpu())
    check_accuracy(s_merged.cpu(), expected_s_merged.cpu())
    check_accuracy(d_merged.cpu(), expected_d_merged.cpu())


if __name__ == "__main__":
    # 编译测试
    path = "state_merge_npu_output.pt"
    save_inputs_outputs(path)

    # 对比cuda和triton-ascend的输出
    path = "state_merge_cuda_output.pt"
    run_and_compare(path)
    # >>> Compare Type: float32
    # Max diff at (tensor(12), tensor(4), tensor(63)): test=2.9325761795043945, ref=2.9325757026672363, abs=4.76837158203125e-07, rel=1.6260005963886215e-07
    # 精度达标 (0/524288, 0.000000% <= 0.010000%)
    # >>> Compare Type: float32
    # 精度达标 (0/8192, 0.000000% <= 0.010000%)
    # >>> Compare Type: float32
    # Max diff at (tensor(102), tensor(0)): test=1.9168145656585693, ref=1.9168143272399902, abs=2.384185791015625e-07, rel=1.2438265173386753e-07
    # 精度达标 (0/8192, 0.000000% <= 0.010000%)
