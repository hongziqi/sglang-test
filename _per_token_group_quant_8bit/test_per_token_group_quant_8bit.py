from typing import Tuple

import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_npu, print_real_data

def is_hip() -> bool:
    return torch.version.hip is not None

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


# 定义自动调优配置
per_token_group_quant_8bit_autotune = triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=[],
    auto_profile_dir="/home/coder/.autotune",
)


@triton.jit
def _per_token_group_quant_8bit(
    y_ptr,         # 原始输入：float16/float32 维度 [M, N]
    y_q_ptr,       # 量化输出：int8 或 float8 类型，shape 同 y_ptr
    y_s_ptr,       # 存储每个 group 的缩放因子，shape = [M]
    y_stride,      # 行步长
    N,             # group 内元素数量
    eps,           # 极小值，防止除以0
    fp8_min,      # 8bit 数据的最大值
    fp8_max,      # 8bit 数据的最小值
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
    y_s = _absmax / fp8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


_per_token_group_quant_8bit_autotune = per_token_group_quant_8bit_autotune(_per_token_group_quant_8bit)


def triton_per_token_group_quant_8bit_impl(
    x: torch.Tensor,        # 输入张量，shape [batch_size * seq_len, hidden_dim]
    x_q: torch.Tensor,     # 量化输出张量，shape 同 x
    x_s: torch.Tensor,     # 存储每个 group 的缩放因子，shape = [batch_size * seq_len, hidden_dim // group_size]
    group_size: int,        # 每组的元素数量
    N: int,             # group 内元素数量
    eps: float,     # 防止除以0的极小值
    fp8_min,
    fp8_max,
    BLOCK: int,  # Triton 线程块大小
    num_warps: int,  # 每个线程块的 warp 数量
    num_stages: int,  # 每个线程块的阶段数
    autotune: bool = False,  # 是否自动调优
    **kwargs,
):
    M, N = x.shape
    assert N % group_size == 0, "N must be divisible by group_size"
    if autotune:
        _per_token_group_quant_8bit_autotune[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
        )
    else:
        _per_token_group_quant_8bit[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,        # 输入张量，shape [batch_size * seq_len, hidden_dim]
    group_size: int,        # 每组的元素数量
    dst_dtype: torch.dtype, # 目标数据类型（int8 或 fp8）
    eps: float = 1e-10,     # 防止除以0的极小值
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
    assert x.is_contiguous(), "`x` is not contiguous"

    if dst_dtype == torch.int8:
        finfo = torch.iinfo(dst_dtype)
    else:
        finfo = torch.finfo(dst_dtype)

    fp8_max = finfo.max

    if _is_hip:
        if dst_dtype == torch.int8:
            fp8_max = 127.0
        else:
            fp8_max = 224.0

    fp8_min = -fp8_max
    # print("fp8_min:", fp8_min)
    # print("fp8_max:", fp8_max)
    
    # Prepare output tensors
    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_8bit[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_q, x_s


def save_inputs_outputs(path, batch_size=1, seq_len=64, hidden_dim=7168, group_size=128, dst_type=torch.int8, eps=1e-10):
    # 输入规格
    # x (batch_size * seq_len, hidden_dim) float32 原始输入
    # y_q (batch_size * seq_len, hidden_dim) int8/fp8_type_ 量化输出
    # y_s (batch_size * seq_len, hidden_dim // group_size) float32 存储每个 group 的缩放因子
    x = torch.randn((batch_size * seq_len, hidden_dim), dtype=torch.float32, device="npu")
    y_q, y_s = triton_per_token_group_quant_8bit(x, group_size, dst_type, eps)

    # print(">>x.shape:", x.shape)
    # print(">>y_q.shape:", y_q.shape)
    # print(">>y_s.shape:", y_s.shape)
    
    torch.save({
        "x": x.cpu(),
        "y_q": y_q.cpu(),
        "y_s": y_s.cpu(),
        "group_size": group_size,
        "eps": eps,
        "dst_type": dst_type,
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

    y_q, y_s = triton_per_token_group_quant_8bit(x, group_size, torch.int8, eps)

    expected_y_q = data["y_q"].to("npu")
    expected_y_s = data["y_s"].to("npu")

    check_accuracy(y_q, expected_y_q)
    check_accuracy(y_s, expected_y_s)


if __name__ == "__main__":
    # 1. 编译测试
    # path = "per_token_group_quant_8bit_npu_output.pt"
    # save_inputs_outputs(path) # int8 编译成功
    # # save_inputs_outputs(path, dst_type=fp8_type_) # fp8 不支持
    
    # 2.对比cuda和triton-ascend的输出
    # path = "per_token_group_quant_8bit_cuda_output.pt"
    # run_and_compare(path)

    # 3.对比真实数据并检查精度
    key_mapping = {
        "x": "x",
        "x_q": "x_q",
        "x_s": "x_s",
        "group_size": "group_size",
        "N": "N",
        "fp8_min": "fp8_min",
        "fp8_max": "fp8_max",
        "eps": "eps",
        "dst_type": "dst_type",
        "BLOCK": "BLOCK",
    }
    accuracy_dict = ["x_q", "x_s"]
    src_path = "105_per_token_group_quant_fp8_debug.pt"
    expected_path = "105_per_token_group_quant_fp8_expected.pt"
    run_and_compare_real_data_npu(
        triton_kernel_impl=triton_per_token_group_quant_8bit_impl,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        accuracy=True,  # 检查精度
        accuracy_dict=accuracy_dict,
    )

    # 4.1 测试 autotune kernel 的性能 (BLOCK_SIZE:)
    # run_and_compare_real_data_npu(
    #     triton_kernel_impl=triton_per_token_group_quant_8bit_impl,
    #     src_path=src_path,
    #     expected_path=expected_path,
    #     key_mapping=key_mapping,
    #     accuracy=False,
    #     autotune=True,  # 自动调优
    #     profiling=True,  # 启用性能分析
    # )

    # # 4.2 测试 normal kernel 的性能 (BLOCK_SIZE:128)
    # run_and_compare_real_data_npu(
    #     triton_kernel_impl=triton_per_token_group_quant_8bit_impl,
    #     src_path=src_path,
    #     expected_path=expected_path,
    #     key_mapping=key_mapping,
    #     accuracy=False,
    #     autotune=False,  # 不自动调优
    #     profiling=True,  # 启用性能分析
    # )
