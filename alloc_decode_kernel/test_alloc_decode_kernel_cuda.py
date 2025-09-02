import torch
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_cuda, print_real_data

# 定义自动调优配置
alloc_decode_autotune = triton.autotune(
    configs=[
        triton.Config({'page_size': 8}),
        triton.Config({'page_size': 16}),
        triton.Config({'page_size': 32}),
        triton.Config({'page_size': 64}),
        triton.Config({'page_size': 128}),
        triton.Config({'page_size': 256}),
        triton.Config({'page_size': 512}),
    ],
    key=[],
)

@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    ret_values,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Return value
    if pid == tl.num_programs(0) - 1:
        tl.store(ret_values, sum_num_new_pages)

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1

alloc_decode_kernel_autotuned = alloc_decode_autotune(alloc_decode_kernel)

def alloc_decode_triton_launcher(
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    ret_values,
    page_size,
    autotune: bool = False, # 是否自动调优
):
    bs = len(seq_lens)
    grid = (bs,)

    if autotune:
        alloc_decode_kernel_autotuned[grid](
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            ret_values,
            bs_upper=bs,
        )
    else:
        alloc_decode_kernel[grid](
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            ret_values,
            bs_upper=bs,
            page_size=page_size,
        )


if __name__ == "__main__":
    # 1. 运行真实数据，并保存运行结果
    # [REAL DATA INFO]
    # >> seq_lens:
    # Shape: torch.Size([1])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [12]
    # >> last_loc:
    # Shape: torch.Size([1])
    # Dtype: torch.int32
    # Device: cpu
    # First 10 elements: [18]
    # >> free_pages:
    # Shape: torch.Size([7498])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # >> out_indices:
    # Shape: torch.Size([1])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [10]
    # >> ret_values:
    # Shape: torch.Size([])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [0]
    # >> bs_padded: 1
    # >> page_size: 8
    key_mapping = {
        "seq_lens": "seq_lens",
        "last_loc": "last_loc",
        "free_pages": "free_pages",
        "out_indices": "out_indices",
        "ret_values": "ret_values",
        "bs_padded": "bs_padded",
        "page_size": "page_size",
    }
    src_path = "88_alloc_decode_kernel_debug_cuda0.pt"
    expected_path = "88_alloc_decode_kernel_expected_cuda0.pt"
    # 1. 运行真实数据，并保存运行结果
    run_and_compare_real_data_cuda(
        triton_kernel_impl=alloc_decode_triton_launcher,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        save_output=True,   # 保存运行结果
    )

    # 2. 测试autotune kernel的性能(真实数据)
    run_and_compare_real_data_cuda(
        triton_kernel_impl=alloc_decode_triton_launcher,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        save_output=True,   # 保存运行结果
        autotune=True,  # 使用自动调优
        profiling=True,  # 进行性能分析
    )