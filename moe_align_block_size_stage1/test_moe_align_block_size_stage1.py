import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,       # 表示每个 token 被分配到的 expert ID（范围在 [0, num_experts)）
    tokens_cnts_ptr,    # 用于存储每个线程对每个 expert 的 token 计数
    num_experts: tl.constexpr,  # expert 总数
    numel: tl.constexpr,        # token 总数
    tokens_per_thread: tl.constexpr, # 每个线程处理的 token 数量
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)         # 当前 token 对应的 expert_id
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)   # 获取当前 expert 的计数
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1) # 增加计数


def moe_align_block_size_stage1_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
):
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )

    return tokens_cnts


def save_inputs_outputs(path, num_tokens=1024, num_experts=16, topk=4):
    # topk_ids  [num_tokens, topk] int32，表示每个 token 被分配到的 top-k 个 expert ID
    # tokens_cnts [num_experts + 1, num_experts] int32，每一行是线程分配的 token 计数
    topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32).npu()
    tokens_cnts = moe_align_block_size_stage1_impl(topk_ids, num_experts)

    torch.save({
        "topk_ids": topk_ids.cpu(),
        "tokens_cnts": tokens_cnts.cpu(),
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "topk": topk,
    }, path)


def run_and_compare(path):
    data = torch.load(path)
    topk_ids = data["topk_ids"].npu()
    num_experts = data["num_experts"]

    tokens_cnts = moe_align_block_size_stage1_impl(topk_ids, num_experts)

    expected_tokens_cnts = data["tokens_cnts"]

    print(">> Topk IDs:", topk_ids)
    print(">> Tokens counts:", tokens_cnts)
    print(">> Expected tokens counts:", expected_tokens_cnts)

    check_accuracy(tokens_cnts.cpu(), expected_tokens_cnts.cpu())


if __name__ == "__main__":
    # 编译测试
    # path = "moe_align_block_size_stage1_npu_output.pt"
    # save_inputs_outputs(path)

    # 对比cuda和triton-ascend的输出
    path = "moe_align_block_size_stage1_cuda_output.pt"
    run_and_compare(path)
