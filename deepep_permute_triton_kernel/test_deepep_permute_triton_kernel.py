import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_npu


# 定义自动调优配置
deepep_permute_triton_autotune = triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 64}),
        # triton.Config({'BLOCK_SIZE': 128}),
        # triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=[],
    auto_profile_dir="/home/coder/.autotune",
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
    topk_ids_ptr = topk_ids_ptr + src_idx * topk # 无使用

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
    # grid = (1,)

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


def save_inputs_outputs(path):
    torch.manual_seed(42)
    src_len, dst_len, hidden_size, topk = 8, 16, 128, 2
    BLOCK_SIZE = 64
    input = torch.randn(src_len, hidden_size, device="npu", dtype=torch.float32)
    gateup_input = torch.zeros(dst_len, hidden_size, device="npu", dtype=torch.float32)
    src2dst = torch.randperm(dst_len, device="npu")[:src_len * topk].reshape(src_len, topk)
    topk_ids = torch.randint(0, 10, (src_len, topk), device="npu", dtype=torch.int32)
    # a1_scales = torch.rand(src_len, device="npu", dtype=torch.float32)

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


def run_and_compare(path, BLOCK_SIZE: int = 64):
    data = torch.load(path)
    input = data["input"].to("npu")
    src2dst = data["src2dst"].to("npu")
    topk_ids = data["topk_ids"].to("npu")
    expected_output = data["gateup_input"].to("npu")

    gateup_input = torch.zeros_like(expected_output)
    deepep_permute_impl(
        input,
        gateup_input,
        src2dst,
        topk_ids,
        None,
        topk=topk_ids.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    check_accuracy(gateup_input, expected_output)


if __name__ == "__main__":
    # path = "deepep_permute_cuda_output.pt"
    # run_and_compare(path)       # （测试数据）对比cuda和triton-ascend的输出

    key_mapping = {
        "input": "hidden_states",
        "gateup_input": "gateup_input",
        "src2dst": "src2dst",
        "topk_ids": "topk_idx",
        "topk": "router_topk",
    }
    accuracy_dict=["gateup_input"]
    src_path = "deepep_permute_triton_kernel_debug_cuda0.pt"
    expected_path = "deepep_permute_triton_kernel_expected_cuda0.pt"
    # [REAL DATA INFO]
    # >> hidden_states:
    # Shape: torch.Size([6923, 7168])
    # Dtype: torch.bfloat16
    # Device: cpu
    # First 10 elements: [-0.2578125, -0.1298828125, 0.26953125, -0.40234375, 0.2099609375, -0.099609375, 0.026123046875, 0.10205078125, -0.345703125, -0.01611328125]
    # >> gateup_input:
    # Shape: torch.Size([17393, 7168])
    # Dtype: torch.bfloat16
    # Device: cpu
    # First 10 elements: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # >> src2dst:
    # Shape: torch.Size([55384])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [-37991, -37990, -37989, -37988, -37987, -37986, 10590, -37985, -37984, -37983]
    # >> topk_idx:
    # Shape: torch.Size([6923, 8])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [-1, -1, -1, -1, -1, -1, 20, -1, -1, -1]
    # >> router_topk: 8

    # 3.对比真实数据并检查精度
    run_and_compare_real_data_npu(
        triton_kernel_impl=deepep_permute_impl,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        accuracy=True,  # 是否检查精度
        accuracy_dict=accuracy_dict,
        USE_BLOCK_SIZE=True, # 使用自定义block_size, 非autotune情况下生效
        block_size=128,      # BLOCK_SIZE 设置
    )

    # 4.1 测试 autotune kernel 的性能
    # run_and_compare_real_data_npu(
    #     triton_kernel_impl=deepep_permute_impl,
    #     src_path=src_path,
    #     key_mapping=key_mapping,
    #     accuracy=False,  # 是否检查精度
    #     autotune=True,  # 使用自动调优
    #     profiling=True,  # 进行性能分析
    # )
