import torch
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

def is_hip() -> bool:
    return torch.version.hip is not None
_is_hip = is_hip()


# gelu on first half of vector
@triton.jit
def gelu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    out_scales_ptr,  # (bs,) 用于量化的 scale （未实现）
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    quant_max: tl.constexpr, # 量化上界（未实现）
    static_scale: tl.constexpr, # 是否使用静态 scale（未实现）
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # gelu
    # cast down before mul to better match training?
    # x1 = hidden_states[:, :hidden_dim]：对前一半做 GELU
    gelu_x1 = 0.5 * (1.0 + tl.erf(x1 * 0.7071067811865475)) * x1
    # x3 = hidden_states[:, hidden_dim:]：和 GELU 后的结果做逐元素乘法
    # out = x3 * gelu_x1，[batch_size, hidden_dim]，保存在 out_hidden_states 中
    out = x3 * gelu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def gelu_and_mul_triton(
    hidden_states,
    scales=None,
    quantize=None,  # dtype to quantize to
    out=None,
):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2

    if out is None:
        out_hidden_states = torch.empty(
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)
        assert out.dtype == (quantize or hidden_states.dtype)
        out_hidden_states = out
    out_scales = None
    static_scale = False
    if quantize is not None:
        if scales is None:
            out_scales = torch.empty(
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales
            static_scale = True

    max_warps = 16 if _is_hip else 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    # 输入规格
    # hidden_states (bs, hidden_dim * 2) float32，原始输入，前半部分用于 gelu，后半部分用于乘法
    # out_hidden_states (bs, hidden_dim) float32，输出结果
    # out_scales (bs,) float32，用于量化的 scale（kernel 未使用）
    # quant_max：量化上界（未使用）
    # static_scale：是否使用静态 scale（未使用）
    # hidden_dim：输出的 hidden_dim

    gelu_and_mul_kernel[(bs,)](
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,
        static_scale=static_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )

    if quantize is not None:
        return out_hidden_states, out_scales
    else:
        return out_hidden_states, None


def save_inputs_outputs(path, bs=4, hidden_dim=64, quantize=None):
    # kernel未实现量化，quantize 默认为 None，不用改
    # hidden_states [bs, hidden_dim * 2] float32，前半部分用于 gelu，后半部分用于乘法
    hidden_states = torch.randn((bs, hidden_dim * 2), dtype=torch.float32, device="npu")

    out_hidden_states, out_scales = gelu_and_mul_triton(hidden_states, quantize=quantize)

    # 输出结果
    # print(f"out_hidden_states shape:{out_hidden_states.shape}, dtype: {out_hidden_states.dtype}")
    # print("out_hidden_states:", out_hidden_states)
    # print("out_scales:", out_scales)

    torch.save({
        "hidden_states": hidden_states.cpu(),
        "out_hidden_states": out_hidden_states.cpu(),
        "out_scales": out_scales.cpu() if out_scales is not None else None,
        "bs": bs,
        "hidden_dim": hidden_dim,
        "quantize": quantize,
    }, path)


def run_and_compare(path):
    data = torch.load(path)
    hidden_states = data['hidden_states'].npu()
    quantize = data['quantize']

    out_hidden_states, out_scales = gelu_and_mul_triton(hidden_states, quantize=quantize)

    expected_out_hidden_states = data['out_hidden_states'].npu()
    expected_out_scales = data['out_scales'].npu() if data['out_scales'] is not None else None
    check_accuracy(out_hidden_states.cpu(), expected_out_hidden_states.cpu())
    if expected_out_scales is not None:
        check_accuracy(out_scales.cpu(), expected_out_scales.cpu())
    else:
        assert out_scales is None, "out_scales should be None when quantize is None"


if __name__ == "__main__":
    # 编译测试
    path = "gelu_and_mul_kernel_npu_output.pt"
    save_inputs_outputs(path)

    path = "gelu_and_mul_kernel_cuda_output.pt"
    # 对比cuda和triton-ascend的输出
    run_and_compare(path)
    # >>> Compare Type: float32
    # Max diff at (tensor(1), tensor(11)): test=-3.977464199066162, ref=-3.977463960647583, abs=2.384185791015625e-07, rel=5.994234442141533e-08
    # 精度达标 (0/256, 0.000000% <= 0.010000%)
