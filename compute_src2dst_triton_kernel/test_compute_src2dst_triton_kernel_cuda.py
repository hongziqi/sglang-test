import torch
import triton
import triton.language as tl


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
    reorder_ids = torch.arange(num_toks, device="cuda", dtype=torch.int32)
    src2dst = torch.empty((num_toks,), device="cuda", dtype=torch.int32)

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


def run_and_compare(path, atol: float = 1, rtol: float = 1e-3):
    data = torch.load(path)
    reorder_ids = data["reorder_ids"].cuda()
    src2dst = torch.zeros_like(data["src2dst"]).cuda()
    BLOCK_SIZE = data["BLOCK_SIZE"]

    # 重新计算输出
    compute_src2dst_impl(
        reorder_ids=reorder_ids,
        src2dst=src2dst,
        num_toks=reorder_ids.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output_ref = data["src2dst"].cuda()
    is_close = torch.isclose(src2dst, output_ref, atol=atol, rtol=rtol)
    mismatch_idx = torch.nonzero(~is_close)
    print(f"Output consistent: {is_close.all().item()}\nMax difference: {(src2dst - output_ref).abs().max().item()}")
    for idx in mismatch_idx:
        i, j = idx.tolist()
        print(f"[{i}, {j}]: test={src2dst[i, j]}, ref={output_ref[i, j]}, diff={abs(src2dst[i, j] - output_ref[i, j])}")


if __name__ == "__main__":
    path = "compute_src2dst_cuda_output.pt"
    save_inputs_outputs(path, BLOCK_SIZE=64) # >> Compute src2dst: [0 1 2 3 4 5]
    save_inputs_outputs(path, BLOCK_SIZE=128) # >> Compute src2dst: [0 1 2 3 4 5]

    # 运行并比较结果
    run_and_compare(path)
    # >> Compute src2dst: [0 1 2 3 4 5]
    # Output consistent: True
    # Max difference: 0
