import torch
import triton
import triton.language as tl


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    # 对每一列做前缀和
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


def moe_align_block_size_stage2_impl(
    tokens_cnts: torch.Tensor,
    num_experts: int,
):
    grid = (num_experts,)

    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )


def save_inputs_outputs(path, num_experts=16):
    # tokens_cnts [num_experts + 1, num_experts] int32，每一行是线程分配的 token 计数(stage1的输出)
    tokens_cnts = torch.randint(0, 10, (num_experts + 1, num_experts), dtype=torch.int32, device="cuda")
    tokens_cnts_original = tokens_cnts.clone()
    moe_align_block_size_stage2_impl(tokens_cnts, num_experts)

    torch.save({
        'tokens_cnts_original': tokens_cnts_original.cpu(),
        'tokens_cnts': tokens_cnts.cpu(),
        'num_experts': num_experts,
    }, path)


def run_and_compare(path, rtol: float = 1e-3, atol: float = 1e-3):
    data = torch.load(path)
    tokens_cnts = data['tokens_cnts_original'].cuda()
    num_experts = data['num_experts']

    moe_align_block_size_stage2_impl(tokens_cnts, num_experts)

    expected_tokens_cnts = data['tokens_cnts'].cuda()

    assert torch.allclose(tokens_cnts, expected_tokens_cnts, rtol=rtol, atol=atol), \
        f"Tokens counts mismatch: {tokens_cnts} vs {expected_tokens_cnts}"
    
    print("Output consistent: True")


if __name__ == "__main__":
    path = "moe_align_block_size_stage2_cuda_output.pt"
    save_inputs_outputs(path)
    run_and_compare(path)
