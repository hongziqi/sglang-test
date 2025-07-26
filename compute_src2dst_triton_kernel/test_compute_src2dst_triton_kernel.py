import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy


@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)


def compute_src2dst_impl(
    reorder_ids: torch.Tensor,  # (num_toks,)
    src2dst: torch.Tensor,      # (num_toks, num_toks)
    num_toks: int,              # Total number of tokens
    BLOCK_SIZE: int = 512,      # Block size for processing
):
    grid = lambda meta:((num_toks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    compute_src2dst_triton_kernel[grid](
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_toks=num_toks,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def save_inputs_outputs(path: str, num_toks: int = 6, BLOCK_SIZE: int = 64):
    # 创建输入张量
    reorder_ids = torch.arange(num_toks, device="npu", dtype=torch.int32)
    src2dst = torch.empty((num_toks,), device="npu", dtype=torch.int32)

    # 调用 Triton 内核
    compute_src2dst_impl(
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_toks=num_toks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    print(">> Compute src2dst:", src2dst.cpu().numpy())

    # 保存输入输出
    torch.save({
        "reorder_ids": reorder_ids.cpu(),
        "src2dst": src2dst.cpu(),
        "BLOCK_SIZE": BLOCK_SIZE,
    }, path)


def run_and_compare(path):
    data = torch.load(path)
    reorder_ids = data["reorder_ids"].to("npu")
    src2dst = torch.zeros_like(data["src2dst"]).to("npu")
    BLOCK_SIZE = data["BLOCK_SIZE"]

    # 重新计算输出
    compute_src2dst_impl(
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_toks=reorder_ids.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    print(">> reorder_ids:", reorder_ids.cpu().numpy())
    print(">> Block Size:", BLOCK_SIZE)
    print(">> Compute src2dst:", src2dst.cpu().numpy())


    # 检查结果
    expected_output = data["src2dst"].to("npu")
    print(">> Expected output:", expected_output.cpu().numpy())
    check_accuracy(src2dst, expected_output)


def run_and_compare_real_data(src_path, expected_path):
    """
    [SRC2DST KERNEL REAL DATA]
    >>reorder_ids:
    Shape: torch.Size([1280])
    Dtype: torch.int64
    Device: cpu
    First 10 elements: [1, 9, 17, 25, 33, 41, 49, 57, 65, 73]
    >>src2dst:
    Shape: torch.Size([1280])
    Dtype: torch.int32
    Device: cpu
    First 10 elements: [114, 0, 45, 0, 99, 0, 46, 0, 98, 0]
    >>numel: 1280
    >>BLOCK_SIZE: 512
    >>grid: (3,)
    """
    try:
        data = torch.load(src_path, map_location=torch.device('cpu'))
        expected_data = torch.load(expected_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"File {src_path} or {expected_path} not found. Please run the test to generate it.")
        return
    print("\n[SRC2DST KERNEL REAL DATA]")

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f">>{key}:")
            print(f"  Shape: {value.cpu().shape}")
            print(f"  Dtype: {value.cpu().dtype}")
            print(f"  Device: {value.cpu().device}")
            # 打印前10个元素
            print(f"  First 10 elements: {value.cpu().flatten()[:10].tolist()}")
        else:
            print(f">>{key}: {value}")
    
    reorder_ids = data["reorder_ids"].npu()
    src2dst = data["src2dst"].npu()
    BLOCK_SIZE = data["BLOCK_SIZE"]
    num_toks = data["numel"]

    compute_src2dst_impl(
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_toks=num_toks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    expected_output = expected_data["src2dst"].npu()

    check_accuracy(src2dst, expected_output)


## zhanpeng testcase
def test_compute_src2dst_triton_no_conflict():
    import numpy as np

    render_ids = torch.tensor([0, 1, 2, 3, 4, 5], device="npu", dtype=torch.int32)
    num_toks = render_ids.shape[0]
    BLOCK_SIZE = 128
    src2dst = torch.zeros_like(render_ids).to("npu")

    grid = lambda meta: ((num_toks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    compute_src2dst_triton_kernel[grid](
        reorder_ids=render_ids,
        src2dst=src2dst,
        num_toks=num_toks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    print("reorder_ids:", render_ids)
    print("src2dst:", src2dst)

    excepted = render_ids.argsort().int().cpu().numpy()
    actual = src2dst.cpu().numpy()

    assert np.array_equal(excepted, actual), f"Expected {excepted}, but got {actual}"
    print("Test passed!")


if __name__ == "__main__":
    # 1. 编译测试
    # path = "compute_src2dst_npu_output.pt"
    # save_inputs_outputs(path, BLOCK_SIZE=64) # >> Compute src2dst: [0 1 2 3 4 5]
    # save_inputs_outputs(path, BLOCK_SIZE=128) # >> Compute src2dst: [0 0 1 2 3 4]
    ## TRITON_INTERPRET=1 python test_compute_src2dst_triton_kernel.py  >> Compute src2dst: [0 1 2 3 4 5]

    # 2.1 对比triton-ascend和cuda的输出
    # path = "compute_src2dst_cuda_output.pt"
    # run_and_compare(path)
    # >> reorder_ids: [0 1 2 3 4 5]
    # >> Block Size: 128
    # >> Compute src2dst: [0 0 1 2 3 4]
    # >> Expected output: [0 1 2 3 4 5]
    # >>> Compare Type: int32
    # Max diff at (tensor(1, device='npu:0'),): test=0, ref=1, abs=1, rel=0.9999990463256836
    # 精度不达标 (5/6, 83.333333% > 0.100000%)
    # (1,): test=0.000000, ref=1.000000, diff=1.000000, rel=0.999999
    # (2,): test=1.000000, ref=2.000000, diff=1.000000, rel=0.500000
    # (3,): test=2.000000, ref=3.000000, diff=1.000000, rel=0.333333
    # (4,): test=3.000000, ref=4.000000, diff=1.000000, rel=0.250000
    # (5,): test=4.000000, ref=5.000000, diff=1.000000, rel=0.200000

    # 2.2 对比cuda和triton-ascend的输出(zhanpeng testcase)
    # test_compute_src2dst_triton_no_conflict()
    # reorder_ids: tensor([0, 1, 2, 3, 4, 5], device='npu:0', dtype=torch.int32)
    # src2dst: tensor([0, 0, 1, 2, 3, 4], device='npu:0', dtype=torch.int32)
    # [W716 03:39:22.793892537 compiler_depend.ts:26] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
    # Traceback (most recent call last):
    # File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 132, in <module>
    #     test_compute_src2dst_triton_no_conflict()
    # File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 107, in test_compute_src2dst_triton_no_conflict
    #     assert np.array_equal(excepted, actual), f"Expected {excepted}, but got {actual}"
    # AssertionError: Expected [0 1 2 3 4 5], but got [0 0 1 2 3 4]
    # [ERROR] 2025-07-16-03:39:23 (PID:67079, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception

    # 3. 对比真实数据
    src_path = "src2dst_kernel_debug_cuda0.pt"
    expected_path = "src2dst_kernel_expected_cuda0.pt"
    run_and_compare_real_data(src_path, expected_path)
