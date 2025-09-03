
import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_npu


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


def get_target_cache_loc_impl(
    tgt_cache_loc: torch.Tensor,      # (sum(accept_length) + bs,)
    to_free_slots: torch.Tensor,      # (sum(to_free_num_slots),)
    accept_length: torch.Tensor,      # (bs,)
    to_free_num_slots: torch.Tensor,  # (bs,)
    out_cache_loc: torch.Tensor,      # (bs, num_verify_tokens)
    num_verify_tokens: int,           # Maximum number of tokens to verify
    bs_upper: int = 32,              # Batch size upper bound
    autotune: bool = False,           # 是否自动调优
):

    grid = (bs_upper,)
    get_target_cache_loc[grid](
        tgt_cache_loc,
        to_free_slots,
        accept_length,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens=num_verify_tokens,
        num_verify_tokens_upper=next_power_of_2(num_verify_tokens),
        bs_upper=next_power_of_2(bs_upper),
    )


def run_and_compare(path: str):
    # 模拟数据读取并运行比较cuda和npu结果
    data = torch.load(path)
    tgt_cache_loc = torch.empty_like(data["tgt_cache_loc"]).npu()
    to_free_slots = torch.empty_like(data["to_free_slots"]).npu()
    accept_length = data["accept_length"].npu()
    to_free_num_slots = data["to_free_num_slots"].npu()
    out_cache_loc = data["out_cache_loc"].npu()
    num_verify_tokens = data["num_verify_tokens"]
    bs_upper = data["bs_upper"]
    autotune = False

    get_target_cache_loc_impl(
        tgt_cache_loc,
        to_free_slots,
        accept_length,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
        bs_upper,
        autotune,
    )
    print(">> Compute tgt_cache_loc:", tgt_cache_loc.cpu().numpy())
    print(">> Compute to_free_slots:", to_free_slots.cpu().numpy())

    expected_tgt_cache_loc = data["tgt_cache_loc"].npu()
    expected_to_free_slots = data["to_free_slots"].npu()

    check_accuracy(tgt_cache_loc, expected_tgt_cache_loc)
    check_accuracy(to_free_slots, expected_to_free_slots)


if __name__ == "__main__":
    path = "get_target_cache_loc_cuda_output.pt"
    run_and_compare(path)
    # >> Compute tgt_cache_loc: [  0   1   2   3   4   5   6   7  16  17  18  19  20  32  33  34  35  36
    # 37  38  39  40  41  42  48  49  50  51  52  53  54  55  56  57  64  65
    # 66  67  68  69  70  71  72  73  74  75  76  77  78  80  81  82  83  84
    # 96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 112 113]
    # >> Compute to_free_slots: [  8   9  10  11  12  13  14  15  21  22  23  24  25  26  27  28  29  30
    # 31  43  44  45  46  47  58  59  60  61  62  63  79  85  86  87  88  89
    # 90  91  92  93  94  95 111 114 115 116 117 118 119 120 121 122 123 124
    # 125 126 127]
    # >>> Compare Type: int
    # 精度达标 (Mismatched elements:0/71, 0.000000% <= 0.000000%)
    # >>> Compare Type: int
    # 精度达标 (Mismatched elements:0/57, 0.000000% <= 0.000000%)
