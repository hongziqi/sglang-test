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


rmsnorm_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=8),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=4),
    ],
    key=["hidden_dim"],
)


@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,     # (bs, hidden_dim)
    mid_ptr,        # (bs, hidden_dim)
    activ_ptr,      # (bs, hidden_dim)
    residual_ptr,   # (bs, hidden_dim)
    weight1_ptr,    # (hidden_dim,)
    weight2_ptr,    # (hidden_dim,)
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    "执行了两个 RMSNorm 操作，并在两次归一化之间加上残差（residual）"
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here
        mask=mask,
    )


fused_dual_residual_rmsnorm_kernel_autotune = rmsnorm_autotune(
    fused_dual_residual_rmsnorm_kernel
)


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps, autotune=False):
    assert len(x.shape) == 2
    assert x.shape == residual.shape and x.dtype == residual.dtype
    output, mid = torch.empty_like(x), torch.empty_like(x)
    bs, hidden_dim = x.shape
    if autotune:
        fused_dual_residual_rmsnorm_kernel_autotune[(bs,)](
            output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim
        )
    else:
        max_warps = 16 if _is_hip else 32
        config = {
            "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
            "num_warps": max(
                min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
            ),
        }

        fused_dual_residual_rmsnorm_kernel[(bs,)](
            output,
            mid,
            x,
            residual,
            weight1,
            weight2,
            eps=eps,
            hidden_dim=hidden_dim,
            **config,
        )

    return output, mid


def save_inputs_outputs(path, batch_size=4, hidden_dim=128, eps=1e-6):
    # 数据规格
    # input_tensor (bs, hidden_dim) float32 输入向量
    # residual_tensor (bs, hidden_dim) float32 残差向量
    # weight1_tensor (hidden_dim,) float32 第一次 RMSNorm 的权重
    # weight2_tensor (hidden_dim,) float32 第二次 RMSNorm 的权重
    # output_tensor (bs, hidden_dim) float32 输出向量
    # mid_tensor (bs, hidden_dim) float32 中间向量
    input_tensor = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device="npu")
    residual = torch.randn(batch_size, hidden_dim, dtype=torch.float32, device="npu")
    weight1 = torch.randn(hidden_dim, dtype=torch.float32, device="npu")
    weight2 = torch.randn(hidden_dim, dtype=torch.float32, device="npu")

    output_tensor, mid_tensor = fused_dual_residual_rmsnorm(
        input_tensor, residual, weight1, weight2, eps=eps
    )

    torch.save({
        "input_tensor": input_tensor.cpu(),
        "residual_tensor": residual.cpu(),
        "weight1_tensor": weight1.cpu(),
        "weight2_tensor": weight2.cpu(),
        "output_tensor": output_tensor.cpu(),
        "mid_tensor": mid_tensor.cpu(),
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "eps": eps
    }, path)


def run_and_compare(path):
    data = torch.load(path)
    input_tensor = data["input_tensor"].npu()
    residual_tensor = data["residual_tensor"].npu()
    weight1_tensor = data["weight1_tensor"].npu()
    weight2_tensor = data["weight2_tensor"].npu()
    eps = data["eps"]

    output_tensor, mid_tensor = fused_dual_residual_rmsnorm(
        input_tensor, residual_tensor, weight1_tensor, weight2_tensor, eps=eps
    )

    expected_output = data["output_tensor"].npu()
    expected_mid = data["mid_tensor"].npu()

    print(">> checking output tensor")
    check_accuracy(output_tensor, expected_output)
    print(">> checking mid tensor")
    check_accuracy(mid_tensor, expected_mid)


if __name__ == "__main__":
    # 编译测试
    path = "fused_dual_residual_rmsnorm_kernel_npu_output.pt"
    save_inputs_outputs(path)

    path = "fused_dual_residual_rmsnorm_kernel_cuda_output.pt"
    run_and_compare(path)
    # >> checking output tensor
    # >>> Compare Type: float32
    # Max diff at (tensor(3, device='npu:0'), tensor(95, device='npu:0')): test=14.095657348632812, ref=14.095659255981445, abs=1.9073486328125e-06, rel=1.3531460751892155e-07
    # 精度达标 (0/512, 0.000000% <= 0.010000%)
    # >> checking mid tensor
    # >>> Compare Type: float32
    # Max diff at (tensor(1, device='npu:0'), tensor(33, device='npu:0')): test=4.164679527282715, ref=4.164680004119873, abs=4.76837158203125e-07, rel=1.1449548509290253e-07
    # 精度达标 (0/512, 0.000000% <= 0.010000%)
