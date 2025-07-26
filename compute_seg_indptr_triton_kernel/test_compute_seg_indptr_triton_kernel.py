import torch
import torch_npu
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


# zhanpeng testcases
def test_compute_seg_indptr_triton():
    # 输入数据（必须已排序）
    reorder_topk_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int32, device="npu")
    num_toks = reorder_topk_ids.shape[0]

    num_experts = 3

    seg_indptr = torch.zeros(num_experts + 1, dtype=torch.int32, device="npu")

    grid = lambda meta: (num_experts,)
    compute_seg_indptr_triton_kernel[grid](reorder_topk_ids, seg_indptr, num_toks)

    seg_indptr_cpu = seg_indptr.cpu().numpy()
    print("Computed seg_indptr:", seg_indptr_cpu)

    expected = [0,2,5,7]
    assert all(seg_indptr_cpu == expected), f"Expected {expected}, got{seg_indptr_cpu}"
    print("Test Passed!")


def save_inputs_outputs(path: str, num_toks: int = 8, num_experts: int = 3):
    # 初始化输入张量
    reorder_topk_ids = torch.zeros(num_toks, dtype=torch.int32, device="npu")
    seg_indptr = torch.zeros(num_experts + 1, dtype=torch.int32, device="npu")

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
    # 打印结果
    print("Computed seg_indptr:", seg_indptr.cpu().numpy())

    # 保存输入输出
    torch.save({
        "reorder_topk_ids": reorder_topk_ids.cpu(),
        "seg_indptr": seg_indptr.cpu(),
    }, path)

def run_and_compare(path):
    data = torch.load(path)
    reorder_topk_ids = data["reorder_topk_ids"].to("npu")
    seg_indptr = torch.zeros(data["seg_indptr"].shape, dtype=torch.int32, device="npu")

    # 重新计算输出
    compute_seg_indptr_impl(
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=reorder_topk_ids.shape[0],
    )

    # 检查结果
    expected_output = data["seg_indptr"].to("npu")
    check_accuracy(seg_indptr, expected_output)


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
        expected_data = torch.load(expected_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading data from {src_path}: {e}")
        return

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f">>{key}:")
            print(f" Shape: {value.shape}")
            print(f" Dtype: {value.dtype}")
            print(f" Device: {value.device}")
            # 打印前10个元素
            print(f" First 10 elements: {value.flatten()[:10].tolist()}")
        elif isinstance(value, int):
            print(f">>{key}: {value}")
        else:
            print(f">>{key}: {value}")
    
    reorder_topk_ids = data["reorder_topk_ids"].npu()
    seg_indptr = data["seg_indptr"].npu()
    numel = data["numel"]
    num_experts = data["num_experts"]
    # 重新计算输出
    compute_seg_indptr_impl(
        reorder_topk_ids=reorder_topk_ids,
        seg_indptr=seg_indptr,
        num_toks=numel,
    )
    expected_output = expected_data["seg_indptr"].npu()

    check_accuracy(seg_indptr, expected_output)


if __name__ == "__main__":
    # 1.编译测试
    # path = "compute_seg_indptr_npu_output.pt"
    # save_inputs_outputs(path)
    # Computed seg_indptr: [0 2 4 8]

    # 2.对比cuda和triton-ascend的输出
    # path = "compute_seg_indptr_cuda_output.pt"
    # run_and_compare(path)
    # >>> Compare Type: int32
    # 精度达标 (0/17, 0.000000% <= 0.100000%)

    # 3.对比真实数据
    src_path = "seg_indptr_kernel_debug_cuda0.pt"
    expected_path = "seg_indptr_kernel_expected_cuda0.pt"
    run_and_compare_real_data(src_path, expected_path)
    # >>> Compare Type: int
    # 精度达标 (0/129, 0.000000% <= 0.000000%)
