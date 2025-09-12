import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,                   # (输入)每个 token 被分配到的 expert ID（范围在 [0, num_experts)）
    sorted_token_ids_ptr,           # (输出) 排序后的 token ID
    expert_ids_ptr,                 # (输出) 每个 block 对应的 expert ID
    tokens_cnts_ptr,                # (输入输出) 每个专家已分配的 token 数量
    cumsum_ptr,                     # (输入) 每个 expert 的 token 计数的前缀和
    num_experts: tl.constexpr,      # expert 总数
    block_size: tl.constexpr,       # 每个 block 的大小
    numel: tl.constexpr,            # token 总数
    tokens_per_thread: tl.constexpr, # 每个线程处理的 token 数量
):
    pid = tl.program_id(0)                      # 当前线程的 ID，即当前expert 的编号
    start_idx = tl.load(cumsum_ptr + pid)       # 当前 expert 的 token 分配起始位置
    end_idx = tl.load(cumsum_ptr + pid + 1)     # 当前 expert 的 token 分配结束位置, [start_idx, end_idx)

    for i in range(start_idx, end_idx, block_size):         # 按照 block_size 步长遍历, 设置每个 block 对应的 expert ID
        tl.store(expert_ids_ptr + i // block_size, pid) 

    start_idx = pid * tokens_per_thread        # 当前线程处理的 token 起始位置，每个线程处理 tokens_per_thread 个 token
    off_t = pid * num_experts                  # 当前线程在 tokens_cnts_ptr 中的偏移位置

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)): # 遍历当前线程处理的 token
        expert_id = tl.load(topk_ids_ptr + i)                       # 获取当前 token 对应的 expert_id
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)    # 获取当前 expert 已分配的 token 数量
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id) # 计算当前 token 在排序后的位置（当前 expert 已分配的 token 数量 + 该 expert 之前所有 expert 的 token 数量）
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)           # 将当前 token ID 存储到排序后的位置
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1) # 更新当前 expert 已分配的 token 数量


def moe_align_block_size_stage4_impl(
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    tokens_cnts: torch.Tensor, # [num_experts + 1, num_experts] int32
    cumsum: torch.Tensor, # [num_experts + 1] int32
    num_experts: int,
    block_size: int,
):
    """
    实现了 MoE 路由后 token 的分组和排序，保证每个 expert 的 token 连续排列，并记录每个 block 属于哪个 expert
    """
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )
    torch.npu.synchronize()


def run_and_compare():
    # 构造极简测试数据
    num_experts = 2
    block_size = 2
    numel = 4  # 4个token

    # 每个token分配的expert ID（前2个给expert 0，后2个给expert 1）
    topk_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32).npu()
    sorted_token_ids = torch.empty_like(topk_ids).npu()  # expected: [0, 1, 2, 3]
    expert_ids = torch.empty((ceil_div(numel, block_size),), dtype=torch.int32).npu()  # expected: [0, 1]
    # tokens_cnts初始值（全0），形状[num_experts+1, num_experts] = [3, 2]
    tokens_cnts = torch.zeros((3, 2), dtype=torch.int32).npu()  # expected: [[2,0],[0,2],[0,0]]
    print(">> tokens_cnts initial:", tokens_cnts)
    # cumsum：每个expert的token起始位置（前缀和）
    cumsum = torch.tensor([0, 2, 4], dtype=torch.int32).npu()

    moe_align_block_size_stage4_impl(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )

    print(">> 1.sorted_token_ids:", sorted_token_ids.cpu())
    print(">> 1.expected_sorted_token_ids:", torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    check_accuracy(sorted_token_ids.cpu(), torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    print(">> 2.expert_ids:", expert_ids.cpu())
    print(">> 2.expected_expert_ids:", torch.tensor([0, 1], dtype=torch.int32))
    check_accuracy(expert_ids.cpu(), torch.tensor([0, 1], dtype=torch.int32))
    print(">> 3.tokens_cnts:", tokens_cnts.cpu())
    print(">> 3.expected_tokens_cnts:", torch.tensor([[2,0],[0,2],[0,0]], dtype=torch.int32))
    check_accuracy(tokens_cnts.cpu(), torch.tensor([[2,0],[0,2],[0,0]], dtype=torch.int32))


if __name__ == "__main__":
    # 1.模拟数据测试 NPU 和 GPU 结果是否一致
    run_and_compare()
    # >> 1.sorted_token_ids: tensor([1, 0, 3, 0], dtype=torch.int32)
    # >> 1.expected_sorted_token_ids: tensor([0, 1, 2, 3], dtype=torch.int32)
    # >>> Compare Type: int
    # Max diff at (tensor(3),): test=0, ref=3, abs=3, rel=0.9999997019767761
    # 精度不达标 (Mismatched elements:4/4, 100.000000% > 0.000000%)
    # (0,): test=1.000000, ref=0.000000, diff=1.000000, rel=1000000.000000
    # (1,): test=0.000000, ref=1.000000, diff=1.000000, rel=0.999999
    # (2,): test=3.000000, ref=2.000000, diff=1.000000, rel=0.500000
    # (3,): test=0.000000, ref=3.000000, diff=3.000000, rel=1.000000
    # >> 2.expert_ids: tensor([0, 1], dtype=torch.int32)
    # >> 2.expected_expert_ids: tensor([0, 1], dtype=torch.int32)
    # >>> Compare Type: int
    # 精度达标 (Mismatched elements:0/2, 0.000000% <= 0.000000%)
    # >> 3.tokens_cnts: tensor([[1, 0],
    #         [0, 1],
    #         [0, 0]], dtype=torch.int32)
    # >> 3.expected_tokens_cnts: tensor([[2, 0],
    #         [0, 2],
    #         [0, 0]], dtype=torch.int32)
    # >>> Compare Type: int
    # Max diff at (tensor(0), tensor(0)): test=1, ref=2, abs=1, rel=0.4999997615814209
    # 精度不达标 (Mismatched elements:2/6, 33.333333% > 0.000000%)
    # (0, 0): test=1.000000, ref=2.000000, diff=1.000000, rel=0.500000
    # (1, 1): test=1.000000, ref=2.000000, diff=1.000000, rel=0.500000