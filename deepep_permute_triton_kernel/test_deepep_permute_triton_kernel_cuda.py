import torch
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_cuda, print_real_data


# 定义自动调优配置
deepep_permute_triton_autotune = triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=[],
)


@triton.jit
def deepep_permute_triton_kernel(
    input_ptr,         # input tensor (src_len, hidden_size)
    gateup_input_ptr,  # output tensor (dst_len, hidden_size)
    src2dst_ptr,       # mapping from source to destination indices (src_len, topk)
    topk_ids_ptr,      # top-k expert ids (src_len, topk)
    a1_scales_ptr,     # optional scaling factors (if needed)
    topk, 
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    # Get the current source index
    src_idx = tl.program_id(0)

    # Compute pointers for src2dst and topk_ids
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    # Compute pointer for the source data
    src_ptr = input_ptr + src_idx * hidden_size

    # Process the hidden_size dimension in blocks
    for start_offset in range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        # Load input data for the current block
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)

        # Iterate over the top-k experts
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)  # Load destination index
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


deepep_permute_triton_kernel_autotuned = deepep_permute_triton_autotune(deepep_permute_triton_kernel)


def deepep_permute_impl(
    input: torch.Tensor,          # (src_len, hidden_size)
    gateup_input: torch.Tensor,   # (dst_len, hidden_size)
    src2dst: torch.Tensor,        # (src_len, topk)
    topk_ids: torch.Tensor,       # (src_len, topk)
    a1_scales: torch.Tensor,      # Optional (src_len,)
    topk: int,
    BLOCK_SIZE: int = 512,
    autotune: bool = False,  # 是否自动调优
):
    """
    Perform permutation of input data based on src2dst mapping.

    Args:
        input: Input tensor (src_len, hidden_size).
        gateup_input: Output tensor (dst_len, hidden_size).
        src2dst: Mapping from source to destination indices (src_len, topk).
        topk_ids: Top-k expert ids (src_len, topk).
        a1_scales: Optional scaling factors (src_len,).
        topk: Number of top-k experts.
        hidden_size: Hidden size dimension.
        BLOCK_SIZE: Block size for Triton kernel.
    """
    hidden_size = input.shape[1]
    # assert input.shape[1] == hidden_size
    # assert gateup_input.shape[1] == hidden_size
    # assert src2dst.shape[1] == topk
    # assert topk_ids.shape[1] == topk

    grid = lambda meta: (input.shape[0],)

    if autotune:
        deepep_permute_triton_kernel_autotuned[grid](
            input,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            topk=topk,
            hidden_size=hidden_size,
        )
    else:
        # Launch the Triton kernel
        deepep_permute_triton_kernel[grid](
            input,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            topk=topk,
            hidden_size=hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def save_inputs_outputs(path: str, src_len: int = 8, dst_len: int = 16, hidden_size: int = 128, topk: int = 2, BLOCK_SIZE: int = 64):
    torch.manual_seed(42)
    input = torch.randn(src_len, hidden_size, device="cuda", dtype=torch.float32)
    gateup_input = torch.zeros(dst_len, hidden_size, device="cuda", dtype=torch.float32)
    src2dst = torch.randperm(dst_len, device="cuda")[:src_len * topk].reshape(src_len, topk)
    topk_ids = torch.randint(0, 10, (src_len, topk), device="cuda", dtype=torch.int32)
    # a1_scales = torch.rand(src_len, device="cuda", dtype=torch.float32)

    deepep_permute_impl(
        input,
        gateup_input,
        src2dst,
        topk_ids,
        None,
        topk=topk,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch.save({
        "input": input.cpu(),
        "src2dst": src2dst.cpu(),
        "topk_ids": topk_ids.cpu(),
        "gateup_input": gateup_input.cpu(),
    }, path)


def run_and_compare(path: str, rtol=1e-3, atol=1e-3, BLOCK_SIZE: int = 64):
    torch.set_printoptions(threshold=float('inf'))
    data = torch.load(path)
    input = data["input"].cuda()
    src2dst = data["src2dst"].cuda()
    topk_ids = data["topk_ids"].cuda()
    output_ref = data["gateup_input"].cuda()
    gateup_input = torch.zeros_like(output_ref)

    deepep_permute_impl(
        input,
        gateup_input,
        src2dst,
        topk_ids,
        None,
        topk=topk_ids.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    is_close = torch.isclose(gateup_input, output_ref, rtol=rtol, atol=atol)
    mismatch_idx = torch.nonzero(~is_close)
    print(f"Output consistent: {is_close.all().item()}\nMax difference: {(gateup_input - output_ref).abs().max().item()}")
    for idx in mismatch_idx:
        i, j = idx.tolist()
        print(f"[{i}, {j}]: test={gateup_input[i, j]}, ref={output_ref[i, j]}, diff={abs(gateup_input[i, j] - output_ref[i, j])}")


if __name__ == "__main__":
    # 1.保存输入输出
    save_inputs_outputs("deepep_permute_cuda_output.pt")
    # 加载输入并比较重复输入输出精度
    run_and_compare("deepep_permute_cuda_output.pt")

    # 2. 运行真实数据，并保存运行结果
    # [REAL DATA INFO]
    # >> hidden_states:
    # Shape: torch.Size([5, 7168])
    # Dtype: torch.bfloat16
    # Device: cpu
    # First 10 elements: [0.038818359375, 0.0771484375, 0.1005859375, -0.022705078125, -0.047119140625, -0.1689453125, 0.2236328125, -0.01318359375, 0.053955078125, 0.00830078125]
    # >> gateup_input:
    # Shape: torch.Size([9, 7168])
    # Dtype: torch.bfloat16
    # Device: cpu
    # First 10 elements: [-0.000652313232421875, 0.00070953369140625, -0.00048828125, -0.0013275146484375, 8.535385131835938e-05, -0.0001468658447265625, 0.0003814697265625, 0.0009765625, -0.0027008056640625, 0.0001850128173828125]
    # >> src2dst:
    # Shape: torch.Size([40])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [0, -31, -30, -29, -28, -27, -26, -25, 1, -24]
    # >> topk_idx:
    # Shape: torch.Size([5, 8])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [0, -1, -1, -1, -1, -1, -1, -1, 3, -1]
    # >> router_topk: 8
    key_mapping = {
        "input": "hidden_states",
        "gateup_input": "gateup_input",
        "src2dst": "src2dst",
        "topk_ids": "topk_idx",
        "topk": "router_topk",
    }
    src_path = "deepep_permute_triton_kernel_debug_cuda0.pt"
    expected_path = "deepep_permute_triton_kernel_expected_cuda0.pt"
    run_and_compare_real_data_cuda(
        triton_kernel_impl=deepep_permute_impl,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        save_output=True,   # 保存运行结果
    )
