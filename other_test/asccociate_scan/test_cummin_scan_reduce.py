import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl


@triton.jit
def get_dtype_max(dtype: tl.constexpr):
    """get a value which is greater that all other values of that dtype"""
    # extract the tl.dtype from tl.constexpr so as to use its methods
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float("inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2 ** (width - 1) - 1
        return value
    if dtype_.is_int_unsigned():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2**width - 1
        return value


@triton.jit
def is_floating(x):
    promote_to_tensor = x + tl.zeros((1, ), tl.int1)
    return promote_to_tensor.dtype.is_floating()


@triton.jit
def minimum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        equal |= a_isnan and b_isnan

    mask |= equal & (a_index > b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def tl_cummin(input, index, axis=0):
    return tl.associative_scan(
        (input, index), axis, minimum_with_index_tie_break_right
    )


@triton.jit
def tl_min_tie_break_right(input, index, axis=None, keep_dims=False):
    return tl.reduce(
        (input, index),
        axis,
        minimum_with_index_tie_break_right,
        keep_dims=keep_dims,
    )


@triton.jit(do_not_specialize=["n_elements"])
def add_base_min_kernel(
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_indices_ptrs = out_indices + offset
    out_vals = tl.load(out_ptrs, mask=mask)
    out_indices = tl.load(out_indices_ptrs, mask=mask)

    if pid > 0:
        partial_min_ptrs = partial_min + pid - 1
        last_part_min_via_min = tl.load(partial_min_ptrs)
        partial_min_indices_ptrs = partial_min_indices + pid - 1
        last_part_min_index_via_min = tl.load(partial_min_indices_ptrs)

        final_vals = tl.minimum(out_vals, last_part_min_via_min)
        final_indices = tl.where(
            out_vals <= last_part_min_via_min, out_indices, last_part_min_index_via_min
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


@triton.jit(do_not_specialize=["n_elements"])
def scan_part_min_kernel(
    inp,
    out,
    in_indices,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
    USE_OUT_INDICES: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    max_value = get_dtype_max(inp.type.element_ty)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    if tl.constexpr(USE_OUT_INDICES):
        in_indices_ptrs = out_indices + offset
        in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    else:
        in_indices_vals = offset
    result, cummin_indices = tl_cummin(inp_vals, in_indices_vals, axis=0)

    if tl.constexpr(NEED_PARTIAL):
        # tl.min do not support min_indices_tie_break_right
        part_min_via_min, part_min_indices_via_min = tl_min_tie_break_right(
            inp_vals, in_indices_vals, axis=0
        )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummin_indices, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        partial_min_ptrs = partial_min + pid
        tl.store(partial_min_ptrs, part_min_via_min)

        partial_min_indices_ptrs = partial_min_indices + pid
        tl.store(partial_min_indices_ptrs, part_min_indices_via_min)


@triton.jit(do_not_specialize=["n_elements"])
def scan_part_min_kernel_bak(
    inp,
    out,
    in_indices,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    max_value = get_dtype_max(inp.type.element_ty)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    in_indices_ptrs = out_indices + offset
    in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    result, cummin_indices = tl_cummin(inp_vals, in_indices_vals, axis=0)

    # tl.min do not support min_indices_tie_break_right
    part_min_via_min, part_min_indices_via_min = tl_min_tie_break_right(
        inp_vals, in_indices_vals, axis=0
    )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummin_indices, mask=mask)

    partial_min_ptrs = partial_min + pid
    tl.store(partial_min_ptrs, part_min_via_min)

    partial_min_indices_ptrs = partial_min_indices + pid
    tl.store(partial_min_indices_ptrs, part_min_indices_via_min)


def scan_then_fan_col(inp, out, out_indices, n_ele, dtype, use_out_indices=False):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    # need_partial = True if part_num >= 2 else False
    # if need_partial:
    #     partial_min = torch.empty(part_num, dtype=dtype, device=inp.device)
    #     partial_min_indices = torch.empty(
    #         part_num, dtype=torch.int64, device=inp.device
    #     )
    # else:
    #     partial_min = None
    #     partial_min_indices = None
    partial_min = torch.empty(part_num, dtype=dtype, device=inp.device)
    partial_min_indices = torch.empty(
        part_num, dtype=torch.int64, device=inp.device
    )

    grid = (part_num,)
    print(">>> part_num:", part_num)
    # scan_part_min_kernel[grid](
    #     inp,
    #     out,
    #     out_indices,
    #     out_indices,
    #     partial_min,
    #     partial_min_indices,
    #     n_ele,
    #     BLOCK_SIZE,
    #     need_partial,
    #     use_out_indices,
    # )
    scan_part_min_kernel_bak[grid](
        inp,
        out,
        out_indices,
        out_indices,
        partial_min,
        partial_min_indices,
        n_ele,
        BLOCK_SIZE,
    )

    if part_num >= 2:
        scan_then_fan_col(
            partial_min,
            partial_min,
            partial_min_indices,
            part_num,
            dtype,
            use_out_indices=True,
        )
        add_base_min_kernel[grid](
            out, out_indices, partial_min, partial_min_indices, n_ele, BLOCK_SIZE
        )


@triton.jit(do_not_specialize=["part_num"])
def scan_part_min_abc_kernel(
    inp,
    out,
    in_indices,
    out_indices,
    partial_min,
    partial_min_indices,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
    USE_OUT_INDICES: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C

    mask = b_idx < B
    inp_ptrs = inp + offset
    max_value = get_dtype_max(inp.type.element_ty)
    inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    if tl.constexpr(USE_OUT_INDICES):
        in_indices_ptrs = out_indices + offset
        in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    else:
        in_indices_vals = b_idx
    result, cummin_indices = tl_cummin(inp_vals, in_indices_vals, axis=0)

    if tl.constexpr(NEED_PARTIAL):
        # tl.min do not support min_indices_tie_break_right
        part_min_via_min, part_min_indices_via_min = tl_min_tie_break_right(
            inp_vals, in_indices_vals, axis=0
        )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummin_indices, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        partial_min_ptrs = partial_min + part_offset
        tl.store(partial_min_ptrs, part_min_via_min)

        partial_min_indices_ptrs = partial_min_indices + part_offset
        tl.store(partial_min_indices_ptrs, part_min_indices_via_min)


@triton.jit(do_not_specialize=["part_num"])
def add_base_min_abc_kernel(
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)
    out_indices_ptrs = out_indices + offset
    out_indices = tl.load(out_indices_ptrs, mask=mask)

    if pid_b > 0:
        partial_min_ptrs = partial_min + last_part_offset
        last_part_min_via_min = tl.load(partial_min_ptrs)
        partial_min_index_ptrs = partial_min_indices + last_part_offset
        last_part_min_index_via_min = tl.load(partial_min_index_ptrs)

        final_vals = tl.minimum(out_vals, last_part_min_via_min)
        final_indices = tl.where(
            out_vals <= last_part_min_via_min, out_indices, last_part_min_index_via_min
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


def scan_then_fan(inp, out, out_indices, A, B, C, dtype, use_out_indices=False):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    print(">>> part_num:", part_num)
    need_partial = True if part_num >= 2 else False
    if need_partial:
        partial_min = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)
        partial_min_indices = torch.empty(
            A, part_num, C, dtype=torch.int64, device=inp.device
        )
    else:
        partial_min = None
        partial_min_indices = None

    grid = (A, part_num, C)
    scan_part_min_abc_kernel[grid](
        inp,
        out,
        out_indices,
        out_indices,
        partial_min,
        partial_min_indices,
        B,
        C,
        part_num,
        BLOCK_SIZE,
        need_partial,
        use_out_indices,
    )

    if part_num >= 2:
        scan_then_fan(
            partial_min,
            partial_min,
            partial_min_indices,
            A,
            part_num,
            C,
            dtype,
            use_out_indices=True,
        )
        add_base_min_abc_kernel[grid](
            out,
            out_indices,
            partial_min,
            partial_min_indices,
            B,
            C,
            part_num,
            BLOCK_SIZE,
        )


@triton.jit()
def scan_part_min_abc_loop_kernel(
    inp,
    out,
    out_indices,
    B,
    C,
    loop_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_c = tl.program_id(1)

    a_idx = pid_a
    c_idx = pid_c
    t_idx = tl.arange(0, BLOCK_SIZE)
    ac_offset = a_idx * B * C + c_idx

    # init
    max_value = get_dtype_max(inp.type.element_ty)
    if tl.constexpr(inp.type.element_ty.is_fp16()) or tl.constexpr(
        inp.type.element_ty.is_bf16()
    ):
        compute_dtype = tl.float32
    elif tl.constexpr(inp.type.element_ty.is_int8()) or tl.constexpr(
        inp.type.element_ty.is_int16()
    ):
        compute_dtype = tl.int32
    else:
        compute_dtype = inp.type.element_ty

    prev_min_val = tl.full([], max_value, dtype=compute_dtype)
    prev_min_val_idx = tl.full([], 0, dtype=tl.int64)
    last_mask = t_idx == (BLOCK_SIZE - 1)

    for l_idx in tl.range(loop_num):
        b_idx = l_idx * BLOCK_SIZE + t_idx
        mask = b_idx < B
        offset = ac_offset + b_idx * C

        inp_vals = tl.load(inp + offset, mask=mask, other=max_value)
        # Only promote if necessary
        if tl.constexpr(compute_dtype != inp.type.element_ty):
            vals = inp_vals.to(compute_dtype)
        else:
            vals = inp_vals
        idxs = b_idx

        # cummin
        result, cummin_indices = tl_cummin(vals, idxs, axis=0)

        # broadcast
        prev_min_val_b = tl.broadcast_to(prev_min_val, (BLOCK_SIZE,))
        prev_min_val_idx_b = tl.broadcast_to(prev_min_val_idx, (BLOCK_SIZE,))

        # Handle NaN and tie-breaking logic
        if tl.constexpr(compute_dtype.is_floating()):
            # For floats: handle NaN propagation + tie-break right
            prev_is_nan = prev_min_val != prev_min_val
            result_is_nan = result != result
            prev_nan_mask = tl.broadcast_to(prev_is_nan, (BLOCK_SIZE,))

            use_result = result_is_nan | (~prev_nan_mask & (result <= prev_min_val_b))
        else:
            # For integers: simple tie-break right
            use_result = result <= prev_min_val_b

        final_vals = tl.where(use_result, result, prev_min_val_b)
        final_indices = tl.where(use_result, cummin_indices, prev_min_val_idx_b)

        # update global min val and idx
        prev_min_val = tl.sum(tl.where(last_mask, final_vals, 0), axis=0)
        prev_min_val_idx = tl.sum(tl.where(last_mask, final_indices, 0), axis=0)

        # store result
        tl.store(out + offset, final_vals.to(out.type.element_ty), mask=mask)
        tl.store(out_indices + offset, final_indices, mask=mask)


def scan_then_fan_loop(inp, out, out_indices, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B < 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    loop_num = math.ceil(B / BLOCK_SIZE)

    grid = (A, C)
    scan_part_min_abc_loop_kernel[grid](
        inp,
        out,
        out_indices,
        B,
        C,
        loop_num,
        BLOCK_SIZE,
    )


def cummin(
    input,
    dim=1,
    *,
    dtype=None
) -> torch.return_types.cummin:
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    input = input.contiguous()
    K = input.numel() // M // N

    dtype = input.dtype
    if dtype is torch.bool:
        dtype = torch.int64
    out = torch.empty_like(input, dtype=dtype)
    out_indices = torch.empty_like(input, dtype=torch.int64)

    compute_dtype = out.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        print(">>> scan_then_fan_col")
        out_indices = torch.arange(N, device=input.device).reshape(shape)
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif M * K <= 16:
        print(">>> scan_then_fan")
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        print(">>> scan_then_fan_loop")
        scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)
    return out, out_indices

QUICK_MODE = False
REDUCTION_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 40999, 3)]

def test_accuracy_cummin():
    torch.manual_seed(0)
    shape = (1, 2)
    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, device="npu").to(torch.float16)
    print(">>> inp:", inp)
    print(">>> inp.shape:", inp.shape)
    print(">>> dim:", dim)

    ref_out = torch.cummin(inp, dim=-1)
    print(">>> 1. ref_out.values:", ref_out.values)
    print(">>> 1. ref_out.indices:", ref_out.indices)

    res_out = cummin(inp, dim=1)
    print(">>> 2. res_out.values:", res_out[0])
    print(">>> 2. res_out.indices:", res_out[1])
    assert torch.allclose(ref_out.values, res_out[0], atol=1e-2, rtol=1e-2)
    assert torch.equal(ref_out.indices, res_out[1])
    print(">>> test passed!")
    

test_accuracy_cummin()
# Dumping intermediate results to /home/coder/.triton/dump/KVPQBJmqVV3BfP8fd9RFaLNT-XiLiZV5c_nltxwmGoU
# Traceback (most recent call last):
#   File "/home/coder/workspace/scan/triton-ascend/triton/compiler/compiler.py", line 288, in compile
#     next_module = compile_ir(module, metadata)
#   File "/home/coder/workspace/scan/triton-ascend/triton/backends/ascend/compiler.py", line 505, in <lambda>
#     lambda src, metadata: linalg_to_bin_enable_npu_compile(
#   File "/home/coder/workspace/scan/triton-ascend/triton/backends/ascend/compiler.py", line 349, in linalg_to_bin_enable_npu_compile
#     ret = subprocess.run(cmd_list, capture_output=True, check=True)
#   File "/home/coder/miniconda/envs/triton/lib/python3.10/subprocess.py", line 526, in run
#     raise CalledProcessError(retcode, process.args,
# subprocess.CalledProcessError: Command '['/home/coder/shared/bisheng_toolkit_0917//bishengir/bin/bishengir-compile', '/tmp/tmppghr3cx1/kernel.ttadapter.mlir', '--enable-hfusion-compile=true', '--enable-hivm-compile=true', '--enable-triton-kernel-compile=true', '-o', '/tmp/tmppghr3cx1/kernel']' returned non-zero exit status 1.

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/home/coder/workspace/scan/triton-ascend/../../sglang-test/other_test/asccociate_scan/test_cummin_scan_reduce.py", line 587, in <module>
#     test_accuracy_cummin()
#   File "/home/coder/workspace/scan/triton-ascend/../../sglang-test/other_test/asccociate_scan/test_cummin_scan_reduce.py", line 579, in test_accuracy_cummin
#     res_out = cummin(inp, dim=1)
#   File "/home/coder/workspace/scan/triton-ascend/../../sglang-test/other_test/asccociate_scan/test_cummin_scan_reduce.py", line 554, in cummin
#     scan_then_fan_col(input, out, out_indices, N, compute_dtype)
#   File "/home/coder/workspace/scan/triton-ascend/../../sglang-test/other_test/asccociate_scan/test_cummin_scan_reduce.py", line 235, in scan_then_fan_col
#     scan_part_min_kernel_bak[grid](
#   File "/home/coder/workspace/scan/triton-ascend/triton/runtime/jit.py", line 331, in <lambda>
#     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
#   File "/home/coder/workspace/scan/triton-ascend/triton/runtime/jit.py", line 634, in run
#     kernel = self.compile(
#   File "/home/coder/workspace/scan/triton-ascend/triton/compiler/compiler.py", line 297, in compile
#     raise MLIRCompilationError(stage_name, error_detail)
# triton.compiler.errors.MLIRCompilationError: 
# ///------------------[ERROR][Triton][BEG]------------------
# [ConvertLinalgRToBinary] encounters error:
# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":67:18): error: 'hivm.hir.vreduce' op invalid dst index elemtype
# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":67:18): error: 'hivm.hir.vreduce' op invalid dst index elemtype
# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":67:18): error: 'hivm.hir.vreduce' op invalid dst index elemtype
# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":67:18): error: 'hivm.hir.vreduce' op invalid dst index elemtype
# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":1:1): error: Failed to run BiShengHIR pipeline

# loc("/tmp/tmppghr3cx1/kernel.ttadapter.mlir":67:18): error: 'hivm.hir.vreduce' op invalid dst index elemtype
# [ERROR] Failed to run BiShengIR pipeline
# ///------------------[ERROR][Triton][END]------------------

# [ERROR] 2025-09-26-06:25:19 (PID:1597716, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
