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


@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,         # 输出向量 (bs, hidden_dim) 
    activ_ptr,          # 输入向量 (bs, hidden_dim)
    weight_ptr,         # 权重向量 (hidden_dim,)
    eps: tl.constexpr,  # 防止除零的常数
    hidden_dim: tl.constexpr,   # 输入向量的维度
    BLOCK_SIZE: tl.constexpr,   # 每个线程块的大小
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    # rmsnorm(activ) = activ / rms * weight, rms = sqrt(mean(activ^2) + eps)
    a_rms = a / rms * w1

    tl.store(
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps, autotune=False, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_rmsnorm_kernel[(bs,)](
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    return output


def save_inputs_outputs(path, batch_size=4, hidden_dim=128, eps=1e-6):
    # 数据规格
    # input_tensor (bs, hidden_dim) float32
    # weight_tensor (hidden_dim,) float32
    # output_tensor (bs, hidden_dim) float32
    input_tensor = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device='npu')
    weight_tensor = torch.randn(hidden_dim, dtype=torch.float32, device='npu')
    
    output_tensor = fused_rmsnorm(input_tensor, weight_tensor, eps=eps)

    # print(f"Input Tensor: {input_tensor}")
    # print(f"Weight Tensor: {weight_tensor}")
    # print(f"Output Tensor: {output_tensor}")

    # 保存输入输出张量
    torch.save({
        'input_tensor': input_tensor.cpu(),
        'weight_tensor': weight_tensor.cpu(),
        'output_tensor': output_tensor.cpu(),
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'eps': eps
    }, path)


def run_and_compare(path, rtol: float = 1e-3, atol: float = 1e-3):
    data = torch.load(path)
    input_tensor = data['input_tensor'].npu()
    weight_tensor = data['weight_tensor'].npu()
    eps = data['eps']
    
    output_tensor = fused_rmsnorm(input_tensor, weight_tensor, eps=eps)

    # 检查输出张量
    expected_output = data['output_tensor'].npu()
    
    check_accuracy(output_tensor, expected_output)


if __name__ == "__main__":
    # 编译测试
    path = "fused_rmsnorm_kernel_npu_output.pt"
    save_inputs_outputs(path)

    path = "fused_rmsnorm_kernel_cuda_output.pt"
    run_and_compare(path)
    # >>> Compare Type: float32
    # Max diff at (tensor(3, device='npu:0'), tensor(13, device='npu:0')): test=3.1167967319488525, ref=3.1167972087860107, abs=4.76837158203125e-07, rel=1.5298942912522762e-07
    # 精度达标 (0/512, 0.000000% <= 0.010000%)
