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
    return sorted_token_ids, expert_ids, tokens_cnts


def run_and_compare(path):
    data = torch.load(path, map_location='cpu')
    topk_ids = data["topk_ids"].npu()
    sorted_token_ids = torch.empty_like(data["sorted_token_ids"]).npu()
    expert_ids = torch.empty_like(data["expert_ids"]).npu()
    tokens_cnts = torch.empty_like(data["tokens_cnts"]).npu()
    cumsum = data["cumsum"].npu()
    num_experts = data["num_experts"]
    block_size = data["block_size"]

    expected_sorted_token_ids = data["sorted_token_ids"]
    expected_expert_ids = data["expert_ids"]
    expected_tokens_cnts = data["tokens_cnts"]

    sorted_token_ids, expert_ids, tokens_cnts = moe_align_block_size_stage4_impl(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )

    print(">> 1.sorted_token_ids:", sorted_token_ids.cpu())
    print(">> 1.expected_sorted_token_ids:", expected_sorted_token_ids.cpu())
    check_accuracy(sorted_token_ids.cpu(), expected_sorted_token_ids.cpu())
    print(">> 2.expert_ids:", expert_ids.cpu())
    print(">> 2.expected_expert_ids:", expected_expert_ids.cpu())
    check_accuracy(expert_ids.cpu(), expected_expert_ids.cpu())
    print(">> 3.tokens_cnts:", tokens_cnts.cpu())
    print(">> 3.expected_tokens_cnts:", expected_tokens_cnts.cpu())
    check_accuracy(tokens_cnts.cpu(), expected_tokens_cnts.cpu())


if __name__ == "__main__":
    # 1.模拟数据测试 NPU 和 GPU 结果是否一致
    path = "moe_align_block_size_stage4_cuda_output.pt"
    run_and_compare(path)