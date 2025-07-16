import torch
import triton
import triton.language as tl

@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)


# zhanpeng testcases
def test_compute_seg_indptr_triton():
    # 输入数据（必须已排序）
    reorder_topk_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int32, device="cuda")
    num_toks = reorder_topk_ids.shape[0]

    num_experts = 3

    seg_indptr = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")

    grid = lambda meta: (num_experts,)
    compute_seg_indptr_triton_kernel[grid](reorder_topk_ids, seg_indptr, num_toks)

    seg_indptr_cpu = seg_indptr.cpu().numpy()
    print("Computed seg_indptr:", seg_indptr_cpu)

    expected = [0,2,5,7]
    assert all(seg_indptr_cpu == expected), f"Expected {expected}, got{seg_indptr_cpu}"
    print("Test Passed!")


def compute_seg_indptr_impl(
    reorder_topk_ids: torch.Tensor,  # (num_toks,)
    seg_indptr: torch.Tensor,        # (num_experts + 1,)
    num_toks: int,                   # Total number of tokens
):
    num_experts = seg_indptr.shape[0] - 1

    grid = lambda meta: (num_experts,)

    # Launch the Triton kernel
    compute_seg_indptr_triton_kernel[grid](
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=num_toks,
    )


def save_inputs_outputs(path: str, num_toks: int = 8, num_experts: int = 3):
    # 初始化输入张量
    reorder_topk_ids = torch.arange(num_toks, dtype=torch.int32, device="cuda")
    seg_indptr = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")

    # 构造 reorder_topk_ids，模拟每个 token 的 expert id
    for i in range(num_experts):
        start_idx = i * (num_toks // num_experts)
        end_idx = (i + 1) * (num_toks // num_experts)
        reorder_topk_ids[start_idx:end_idx] = i

    compute_seg_indptr_impl(
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=num_toks,
    )

    print("Computed seg_indptr:", seg_indptr.cpu().numpy())

    # 保存输入输出
    torch.save({
        "reorder_topk_ids": reorder_topk_ids.cpu(),
        "seg_indptr": seg_indptr.cpu(),
    }, path)


def run_and_compare(path, atol: float = 1, rtol: float = 1e-3):
    data = torch.load(path)

    reorder_topk_ids = data["reorder_topk_ids"].to("cuda")
    seg_indptr = torch.zeros_like(data["seg_indptr"]).to("cuda")

    # 重新计算输出
    compute_seg_indptr_impl(
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=reorder_topk_ids.shape[0],
    )

    # 检查结果
    output_ref = data["seg_indptr"].to("cuda")
    is_close = torch.isclose(seg_indptr, output_ref, atol=atol, rtol=rtol)
    mismatch_idx = torch.nonzero(~is_close)
    print(f"Output consistent: {is_close.all().item()}\nMax difference: {(seg_indptr - output_ref).abs().max().item()}")
    for idx in mismatch_idx:
        i, j = idx.tolist()
        print(f"[{i}, {j}]: test={seg_indptr[i, j]}, ref={output_ref[i, j]}, diff={abs(seg_indptr[i, j] - output_ref[i, j])}")


if __name__ == "__main__":
    path = "compute_seg_indptr_cuda_output.pt"
    save_inputs_outputs(path, 4096, 15)

    # 运行并比较结果
    run_and_compare(path)
    # Computed seg_indptr: [   0  273  546  819 1092 1365 1638 1911 2184 2457 2730 3003 3276 3549
    # 3822 4095]
    # Output consistent: True
    # Max difference: 0
