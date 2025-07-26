import torch
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

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
    print("reorder_topk_ids:", reorder_topk_ids.cpu().numpy())
    print("num_toks:", num_toks)
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
    reorder_topk_ids = torch.zeros(num_toks, dtype=torch.int32, device="cuda")
    seg_indptr = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")

    # 构造排序的 reorder_topk_ids，模拟每个 token 的 expert id
    for i in range(num_experts):
        start_idx = i * (num_toks // num_experts)
        # 最后一个 expert 的 end_idx 应该是 num_toks
        end_idx = (i + 1) * (num_toks // num_experts) if i < num_experts - 1 else num_toks
        reorder_topk_ids[start_idx:end_idx] = i

    compute_seg_indptr_impl(
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=num_toks,
    )
    print("reorder_topk_ids:", reorder_topk_ids.cpu().numpy())
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


def run_and_compare_real_data(src_path, expected_path):
    """
    [SEG INDPTR KERNEL REAL DATA]
    >>reorder_topk_ids:
    Shape: torch.Size([1280])
    Dtype: torch.int64
    Device: cpu
    First 10 elements: [45, 45, 45, 45, 45, 45, 45, 45, 45, 45]
    >>seg_indptr:
    Shape: torch.Size([129])
    Dtype: torch.int64
    Device: cpu
    First 10 elements: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>numel: 1280
    >>num_experts: 128
    """
    try:
        data = torch.load(src_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"File {src_path} not found. Please run the test to generate it.")
        return
    print("\n[SEG INDPTR KERNEL REAL DATA]")
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
    
    reorder_topk_ids = data["reorder_topk_ids"].cuda()
    seg_indptr = data["seg_indptr"].cuda()
    numel = data["numel"]
    num_experts = data["num_experts"]

    # 重新计算输出
    compute_seg_indptr_impl(
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=numel,
    )

    torch.save({
        "reorder_topk_ids": reorder_topk_ids.cpu(),
        "seg_indptr": seg_indptr.cpu(),
        "numel": numel,
        "num_experts": num_experts,
    }, expected_path)


if __name__ == "__main__":
    # 1. 运行并比较结果
    # path = "compute_seg_indptr_cuda_output.pt"
    # save_inputs_outputs(path, 4096, 16)
    # run_and_compare(path)
    # reorder_topk_ids: [ 0  0  0 ... 15 15 15]
    # Computed seg_indptr: [   0  256  512  768 1024 1280 1536 1792 2048 2304 2560 2816 3072 3328
    # 3584 3840 4096]
    # Output consistent: True
    # Max difference: 0

    # 2. 运行真实数据, 并保存运行结果
    src_path = "seg_indptr_kernel_debug_cuda0.pt"
    expected_path = "seg_indptr_kernel_expected_cuda0.pt"
    run_and_compare_real_data(src_path, expected_path)
