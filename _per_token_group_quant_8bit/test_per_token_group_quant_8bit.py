from typing import Tuple

import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

def is_hip() -> bool:
    return torch.version.hip is not None

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

@triton.jit
def _per_token_group_quant_8bit(
    y_ptr,         # 原始输入：float16/float32 维度 [M, N]
    y_q_ptr,       # 量化输出：int8 或 float8 类型，shape 同 y_ptr
    y_s_ptr,       # 存储每个 group 的缩放因子，shape = [M]
    y_stride,      # 行步长（=N）
    N,             # group 内元素数量（= group_size）
    eps,           # 极小值，防止除以0
    max_8bit,      # 8bit 数据的最大值（127 or max(fp8)）
    min_8bit,      # 8bit 数据的最小值（-128 or min(fp8)）
    BLOCK: tl.constexpr,  # Triton 线程块大小
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.
    This function converts the tensor values into 8bit values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / max_8bit
    y_q = tl.clamp(y / y_s, min_8bit, max_8bit).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,        # 输入张量，shape [batch_size * seq_len, hidden_dim]
    group_size: int,        # 每组的元素数量
    dst_dtype: torch.dtype, # 目标数据类型（int8 或 fp8）
    eps: float = 1e-10,     # 防止除以0的极小值
    BLOCK_SIZE: int = 256,  # Triton 线程块大小
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform per-token-group quantization on a tensor using Triton.
    
    Args:
        x (torch.Tensor): Input tensor of shape [M, N].
        group_size (int): Number of elements in each group.
        dst_dtype (torch.dtype): Destination data type (int8 or fp8).
        eps (float): Small value to prevent division by zero.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scale factors.
    """
    M, N = x.shape
    assert N % group_size == 0, "N must be divisible by group_size"
    
    # Prepare output tensors
    y_q = torch.empty((M, N // group_size), dtype=dst_dtype, device=x.device)
    y_s = torch.empty((M,), dtype=torch.float32, device=x.device)
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK']),)
    
    _per_token_group_quant_8bit[grid](
        y_ptr=x,
        y_q_ptr=y_q,
        y_s_ptr=y_s,
        y_stride=N,
        N=N // group_size,
        eps=eps,
        max_8bit=127 if dst_dtype == torch.int8 else 255,
        min_8bit=-128 if dst_dtype == torch.int8 else 0,
        BLOCK=BLOCK_SIZE,  # Adjust as needed
    )
    
    return y_q, y_s


def save_inputs_outputs(path, batch_size=1, seq_len=64, hidden_dim=7168, group_size=128, dst_type=torch.int8, eps=1e-10, BLOCK_SIZE=256):
    # 输入规格
    # x (batch_size * seq_len, hidden_dim) float32 原始输入
    # y_q (batch_size * seq_len, hidden_dim // group_size) int8/fp8_type_ 量化输出
    # y_s (batch_size * seq_len) float32 存储每个 group 的缩放因子
    x = torch.randn((batch_size * seq_len, hidden_dim), dtype=torch.float32, device="npu")
    y_q, y_s = triton_per_token_group_quant_8bit(x, group_size, dst_type, eps, BLOCK_SIZE)
    
    torch.save({
        "x": x.cpu(),
        "y_q": y_q.cpu(),
        "y_s": y_s.cpu(),
        "group_size": group_size,
        "eps": eps,
        "dst_type": dst_type,
        "BLOCK_SIZE": BLOCK_SIZE,
    }, path)


def run_and_compare(path):
    """
    Load saved inputs and outputs, and compare the results.
    
    Args:
        path (str): Path to the saved tensors.
        atol (float): Absolute tolerance for comparison.
        rtol (float): Relative tolerance for comparison.
    """
    data = torch.load(path)
    x = data["x"].to("npu")
    group_size = data["group_size"]
    eps = data["eps"]
    BLOCK_SIZE = data["BLOCK_SIZE"]

    y_q, y_s = triton_per_token_group_quant_8bit(x, group_size, torch.int8, eps, BLOCK_SIZE)

    expected_y_q = data["y_q"].to("npu")
    expected_y_s = data["y_s"].to("npu")

    check_accuracy(y_q, expected_y_q)
    check_accuracy(y_s, expected_y_s)

    

if __name__ == "__main__":
    # 编译测试
    path = "per_token_group_quant_8bit_npu_output.pt"
    save_inputs_outputs(path) # int8 编译成功
    # save_inputs_outputs(path, dst_type=fp8_type_) # fp8 不支持
    
    path = "per_token_group_quant_8bit_cuda_output.pt"
    run_and_compare(path)
    # >>> Compare Type: int8 | uint8
    # Max diff at (tensor(2, device='npu:0'), tensor(27, device='npu:0')): test=63, ref=0, abs=63, rel=63000000.0
    # 精度达标 (1/3584, 0.027902% <= 0.100000%)
    # >>> Compare Type: float32
    # 精度达标 (0/64, 0.000000% <= 0.010000%)
