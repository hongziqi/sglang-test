import torch
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_cuda


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


def save_inputs_outputs(path: str):
    # 模拟数据测试，保存结果
    bs = 8
    num_verify_tokens = 16
    # 随机生成每个 batch 的 accept_length 和 to_free_num_slots
    accept_length = torch.randint(1, num_verify_tokens - 1, (bs,), dtype=torch.int32, device='cuda')
    to_free_num_slots = num_verify_tokens - (accept_length + 1)  # 保证每行总数不变
    # out_cache_loc 每行模拟 cache 索引
    out_cache_loc = torch.arange(bs * num_verify_tokens, dtype=torch.int64, device='cuda').reshape(bs, num_verify_tokens)
    # 输出 shape
    tgt_cache_loc = torch.empty((accept_length.sum() + bs).item(), dtype=torch.int64, device='cuda')
    to_free_slots = torch.empty(to_free_num_slots.sum().item(), dtype=torch.int64, device='cuda')

    get_target_cache_loc_impl(
        tgt_cache_loc,
        to_free_slots,
        accept_length,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
        bs_upper=bs,
        autotune=False,
    )
    print(">> tgt_cache_loc:", tgt_cache_loc.cpu().numpy())
    print(">> to_free_slots:", to_free_slots.cpu().numpy())

    # 保存输入输出
    torch.save({
        "tgt_cache_loc": tgt_cache_loc.cpu(),
        "to_free_slots": to_free_slots.cpu(),
        "accept_length": accept_length.cpu(),
        "to_free_num_slots": to_free_num_slots.cpu(),
        "out_cache_loc": out_cache_loc.cpu(),
        "num_verify_tokens": num_verify_tokens,
        "bs_upper": bs,
    }, path)


def run_and_compare(path):
    # 读取模拟测试数据，比较测试结果，保证和保存时输出结果一致
    data = torch.load(path)
    tgt_cache_loc = torch.empty_like(data["tgt_cache_loc"]).cuda()
    to_free_slots = torch.empty_like(data["to_free_slots"]).cuda()
    accept_length = data["accept_length"].cuda()
    to_free_num_slots = data["to_free_num_slots"].cuda()
    out_cache_loc = data["out_cache_loc"].cuda()
    num_verify_tokens = data["num_verify_tokens"]
    bs_upper = data["bs_upper"]
    autotune = False
    # 重新计算输出
    get_target_cache_loc_impl(
        tgt_cache_loc,
        to_free_slots,
        accept_length,
        to_free_num_slots,
        out_cache_loc,
        num_verify_tokens,
        bs_upper=bs_upper,
        autotune=autotune,
    )
    expected_tgt_cache_loc = data["tgt_cache_loc"].cuda()
    expected_to_free_slots = data["to_free_slots"].cuda()
    print(">> Recomputed tgt_cache_loc:", tgt_cache_loc.cpu().numpy())
    print(">> Recomputed to_free_slots:", to_free_slots.cpu().numpy())
    check_accuracy(tgt_cache_loc, expected_tgt_cache_loc)
    check_accuracy(to_free_slots, expected_to_free_slots)


if __name__ == "__main__":
    # 模拟数据测试
    path = "get_target_cache_loc_cuda_output.pt"
    save_inputs_outputs(path)
    # >> tgt_cache_loc: [  0   1   2   3   4   5   6   7  16  17  18  19  20  32  33  34  35  36
    # 37  38  39  40  41  42  48  49  50  51  52  53  54  55  56  57  64  65
    # 66  67  68  69  70  71  72  73  74  75  76  77  78  80  81  82  83  84
    # 96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 112 113]
    # >> to_free_slots: [  8   9  10  11  12  13  14  15  21  22  23  24  25  26  27  28  29  30
    # 31  43  44  45  46  47  58  59  60  61  62  63  79  85  86  87  88  89
    # 90  91  92  93  94  95 111 114 115 116 117 118 119 120 121 122 123 124
    # 125 126 127]

    # 重复运行并比较保存结果是否一致
    run_and_compare(path)
