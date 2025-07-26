import torch
import triton
import triton.language as tl
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy

@triton.jit
def pre_reorder_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    use_per_token_if_dynamic: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size

    vec = tl.arange(0, BLOCK_SIZE)

    if a1_scales_ptr is not None and use_per_token_if_dynamic:
        scale = 1.0 / tl.load(a1_scales_ptr + src_idx)

    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            if a1_scales_ptr is not None:
                if not use_per_token_if_dynamic:
                    scale = 1.0 / tl.load(a1_scales_ptr + expert_id - start_expert_id)
            else:
                scale = 1.0

            dst_idx = tl.load(src2dst_ptr + idx)
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + vec
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask).to(tl.float32)
                out_data = (in_data * scale).to(OutDtype)
                tl.store(dst_ptr + offset, out_data, mask=mask)


# zhanpeng testcases
def test_pre_reorder_triton():
    num_tokens = 2
    topk = 2
    hidden_size = 4
    num_experts = 3
    start_expert_id = 0
    end_expert_id = num_experts - 1
    BLOCK_SIZE = 4

    # 构造输入数据
    input_data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0], 
         [5.0, 6.0, 7.0, 8.0]
    ], dtype=torch.float32, device="cuda")

    topk_ids = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.int32, device="cuda")

    src2dst = torch.tensor([
        [0, 2],
        [1, 3]
    ], dtype=torch.int32, device="cuda")

    total_expanded_tokens = num_tokens * topk
    gateup_input = torch.zeros((total_expanded_tokens, hidden_size), dtype=torch.float32, device="cuda")
    a1_scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device="cuda")

    grid = lambda meta: (num_tokens,)
    pre_reorder_triton_kernel[grid](
        input_ptr=input_data,
        gateup_input_ptr=gateup_input,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        a1_scales_ptr=a1_scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        use_per_token_if_dynamic=False,  # 设置为 False 以使用全局缩放
    )
    print("Gateup Input after pre-reorder:")
    print(gateup_input)

    # 手动计算期望值
    excepted_output = np.array([
        [1.0 / 1.0, 2.0 / 1.0, 3.0 / 1.0, 4.0 / 1.0],
        [5.0 / 2.0, 6.0 / 2.0, 7.0 / 2.0, 8.0 / 2.0],
        [1.0 / 2.0, 2.0 / 2.0, 3.0 / 2.0, 4.0 / 2.0],
        [5.0 / 4.0, 6.0 / 4.0, 7.0 / 4.0, 8.0 / 4.0]
    ], dtype=np.float32)

    actual_output = gateup_input.cpu().numpy()

    assert np.allclose(actual_output, excepted_output, atol=1e-3), "Test failed."
    print("Test passed!")


def pre_reorder_impl(
    input_data: torch.Tensor,  # (num_tokens, hidden_size)
    gateup_input: torch.Tensor,  # (num_tokens * topk, hidden_size)
    src2dst: torch.Tensor,  # (num_tokens, topk)
    topk_ids: torch.Tensor,  # (num_tokens, topk)
    a1_scales: torch.Tensor,  # (num_experts,)
    start_expert_id: int,
    end_expert_id: int,
    topk: int,
    hidden_size: int,
    BLOCK_SIZE: int = 512,
    use_per_token_if_dynamic: bool = False,
):
    num_tokens = input_data.shape[0]

    grid = lambda meta: (num_tokens,)

    pre_reorder_triton_kernel[grid](
        input_ptr=input_data,
        gateup_input_ptr=gateup_input,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        a1_scales_ptr=a1_scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        use_per_token_if_dynamic=use_per_token_if_dynamic,  # 设置为 False 以使用全局缩放
    )

    return gateup_input


def save_inputs_outputs(path: str, num_tokens: int = 2, topk: int = 2, hidden_size: int = 4, num_experts: int = 3, start_expert_id: int = 0, end_expert_id: int = 2, BLOCK_SIZE: int = 4):
    # 随机生成输入数据
    input_data = torch.randn((num_tokens, hidden_size), dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(low=start_expert_id, high=end_expert_id + 1, size=(num_tokens, topk), dtype=torch.int32, device="cuda")
    src2dst = torch.randint(low=0, high=num_tokens * topk, size=(num_tokens, topk), dtype=torch.int32, device="cuda")
    gateup_input = torch.zeros((num_tokens * topk, hidden_size), dtype=torch.float32, device="cuda")

    a1_scales = torch.rand((end_expert_id - start_expert_id + 1), dtype=torch.float32, device="cuda")
    gateup_input = pre_reorder_impl(
        input_data=input_data,
        gateup_input=gateup_input,
        src2dst=src2dst,
        topk_ids=topk_ids,
        a1_scales=a1_scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 打印输出结果
    print("Gateup Input after pre-reorder:")
    print(gateup_input.cpu().numpy())

    # 保存输入输出
    torch.save({
        "input_data": input_data.cpu(),
        "gateup_input": gateup_input.cpu(),
        "src2dst": src2dst.cpu(),
        "topk_ids": topk_ids.cpu(),
        "a1_scales": a1_scales.cpu(),
        "hidden_size": hidden_size,
        "start_expert_id": start_expert_id,
        "end_expert_id": end_expert_id,
        "topk": topk,
        "BLOCK_SIZE": BLOCK_SIZE,
    }, path)


def run_and_compare(path: str, atol: float = 1e-3, rtol: float = 1e-3):
    data = torch.load(path)
    input_data = data["input_data"].to("cuda")
    gateup_input = torch.zeros_like(data["gateup_input"]).to("cuda")
    src2dst = data["src2dst"].to("cuda")
    topk_ids = data["topk_ids"].to("cuda")
    a1_scales = data["a1_scales"].to("cuda")
    hidden_size = data["hidden_size"]
    start_expert_id = data["start_expert_id"]
    end_expert_id = data["end_expert_id"]
    topk = data["topk"]
    BLOCK_SIZE = data["BLOCK_SIZE"]

    # 重新计算输出
    gateup_input = pre_reorder_impl(
        input_data=input_data,
        gateup_input=gateup_input,
        src2dst=src2dst,
        topk_ids=topk_ids,
        a1_scales=a1_scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        topk=topk,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 检查结果
    output_ref = data["gateup_input"].to("cuda")
    is_close = torch.isclose(gateup_input, output_ref, atol=atol, rtol=rtol)
    mismatch_idx = torch.nonzero(~is_close)
    print(f"Output consistent: {is_close.all().item()}\nMax difference: {(gateup_input - output_ref).abs().max().item()}")
    for idx in mismatch_idx:
        i, j = idx.tolist()
        print(f"[{i}, {j}]: test={gateup_input[i, j]}, ref={output_ref[i, j]}, diff={abs(gateup_input[i, j] - output_ref[i, j])}")


def run_and_compare_real_data(src_path: str, expected_path: str):
    """
    [SEG INDPTR KERNEL REAL DATA]
    >>hidden_states:
    Shape: torch.Size([160, 2048])
    Dtype: torch.bfloat16
    Device: cpu
    First 10 elements: [-0.03271484375, -0.0002899169921875, 0.017333984375, 0.04833984375, 0.0341796875, 0.0174560546875, -0.033935546875, -0.02734375, -0.03662109375, -0.01385498046875]
    >>gateup_input:
    Shape: torch.Size([1280, 2048])
    Dtype: torch.bfloat16
    Device: cpu
    First 10 elements: [-0.15625, -0.1728515625, 0.28515625, 0.337890625, 0.65625, 0.0390625, -0.07568359375, -0.259765625, -0.185546875, -0.01385498046875]
    >>src2dst:
    Shape: torch.Size([1280])
    Dtype: torch.int32
    Device: cpu
    First 10 elements: [800, 520, 40, 200, 960, 1120, 400, 680, 801, 521]
    >>topk_ids:
    Shape: torch.Size([160, 8])
    Dtype: torch.int64
    Device: cpu
    First 10 elements: [92, 77, 18, 60, 97, 115, 71, 81, 92, 77]
    >>w13_input_scale: None
    >>start_expert_id: 0
    >>end_expert_id: 15
    >>top_k: 8
    >>in_features: 2048
    >>BLOCK_SIZE: 512
    >>use_per_token_if_dynamic: True
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

    hidden_states = data["hidden_states"].to("cuda")
    gateup_input = torch.zeros_like(data["gateup_input"]).to("cuda")
    src2dst = data["src2dst"].to("cuda")
    topk_ids = data["topk_ids"].to("cuda")
    w13_input_scale = data.get("w13_input_scale", None)
    start_expert_id = data["start_expert_id"]
    end_expert_id = data["end_expert_id"]
    top_k = data["top_k"]
    in_features = data["in_features"]
    BLOCK_SIZE = data["BLOCK_SIZE"]
    use_per_token_if_dynamic = data.get("use_per_token_if_dynamic", True)

    # 重新计算输出
    gateup_input = pre_reorder_impl(
        input_data=hidden_states,
        gateup_input=gateup_input,
        src2dst=src2dst,
        topk_ids=topk_ids,
        a1_scales=w13_input_scale,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        topk=top_k,
        hidden_size=in_features,
        BLOCK_SIZE=BLOCK_SIZE,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
    )

    torch.save({
        "hidden_states": hidden_states.cpu(),
        "gateup_input": gateup_input.cpu(),
        "src2dst": src2dst.cpu(),
        "topk_ids": topk_ids.cpu(),
        "w13_input_scale": w13_input_scale.cpu() if w13_input_scale is not None else None,
        "start_expert_id": start_expert_id,
        "end_expert_id": end_expert_id,
        "top_k": top_k,
        "in_features": in_features,
        "BLOCK_SIZE": BLOCK_SIZE,
        "use_per_token_if_dynamic": use_per_token_if_dynamic,
    }, expected_path)


if __name__ == "__main__":
    # 1. 运行并比较结果
    # path = "pre_reorder_cuda_output.pt"
    # save_inputs_outputs(path)
    # run_and_compare(path)
    # Gateup Input after pre-reorder:
    # [[  0.          0.          0.          0.       ]
    # [  0.          0.          0.          0.       ]
    # [  1.2335378  -1.5644802  -0.3946963  -3.5499332]
    # [ -6.534607   29.906172  -29.879107  -24.781317 ]]
    # Output consistent: True
    # Max difference: 0.0

    # 2. 运行真实数据, 并保存运行结果
    src_path = "pre_reorder_kernel_debug_cuda0.pt"
    expected_path = "pre_reorder_kernel_expected_cuda0.pt"
    run_and_compare_real_data(src_path, expected_path)
