import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, run_and_compare_real_data_npu, benchmark_compare_close


# 定义自动调优配置
gelu_and_mul_triton_autotune = triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        # triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=[],
    auto_profile_dir="/home/coder/.autotune",
)

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # gelu & mul & quantize
            # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
            # sqrt(2/pi)
            kAlpha = 0.7978845608028654
            gate_output = (
                0.5
                * gate_output
                * (
                    1
                    + tanh(
                        kAlpha
                        * (
                            gate_output
                            + 0.044715 * gate_output * gate_output * gate_output
                        )
                    )
                )
            )
            gate_output = gate_output.to(InDtype)

            gelu_mul_output = gate_output * up_output * scale
            gelu_mul_output = gelu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, gelu_mul_output, mask=mask)


gelu_and_mul_triton_kernel_autotuned = gelu_and_mul_triton_autotune(gelu_and_mul_triton_kernel)


def gelu_and_mul_triton_launcher(
    gateup_output: torch.Tensor,     # shape: [token_num, hidden_size]
    down_input: torch.Tensor,        # shape: [token_num, hidden_size // 2]
    reorder_topk_ids: torch.Tensor,  # shape: [token_num], 每个 token 对应的 expert id
    scales: torch.Tensor | None,     # shape: [expert_range] 缩放因子
    start_expert_id: int,
    end_expert_id: int,
    hidden_size: int= None,
    BLOCK_SIZE: int = 64,
    autotune: bool = False,  # 是否自动调优
):
    if hidden_size is None:
        hidden_size = gateup_output.shape[1]

    grid = (gateup_output.shape[0],)  # 每个 token 一个 program
    if autotune:
        gelu_and_mul_triton_kernel_autotuned[grid](
            gateup_output=gateup_output,
            down_input=down_input,
            hidden_size=hidden_size,
            reorder_topk_ids=reorder_topk_ids,
            scales=scales,
            start_expert_id=start_expert_id,
            end_expert_id=end_expert_id,
        )
    else:
        gelu_and_mul_triton_kernel[grid](
            gateup_output=gateup_output,
            down_input=down_input,
            hidden_size=hidden_size,
            reorder_topk_ids=reorder_topk_ids,
            scales=scales,
            start_expert_id=start_expert_id,
            end_expert_id=end_expert_id,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def save_inputs_outputs(path: str, token_num: int = 8, hidden_size: int = 128, expert_total: int = 64, start_expert_id: int = 0, end_expert_id: int = 31, BLOCK_SIZE: int = 64):
    # 创建输入张量
    gateup_output = torch.ones((token_num, hidden_size), device="npu", dtype=torch.float32)
    down_input = torch.empty(token_num, hidden_size // 2, device="npu", dtype=torch.float32)

    # 模拟每个 token 对应的 expert id（范围在 start_expert_id 到 end_expert_id 之间）
    reorder_topk_ids = torch.randint(
        low=start_expert_id,
        high=end_expert_id + 1,
        size=(token_num,),
        device="npu",
        dtype=torch.int32,
    )

    # 可选：缩放因子 scales，None 表示不使用 scale
    scales = torch.rand(end_expert_id - start_expert_id + 1, device="npu", dtype=torch.float32)

    # 先计算输出
    gelu_and_mul_triton_launcher(
        gateup_output=gateup_output,
        down_input=down_input,
        reorder_topk_ids=reorder_topk_ids,
        scales=scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 保存输入输出
    torch.save({
        "gateup_output": gateup_output,
        "down_input": down_input,
        "reorder_topk_ids": reorder_topk_ids,
        "scales": scales,
        "hidden_size": hidden_size,
        "start_expert_id": start_expert_id,
        "end_expert_id": end_expert_id,
    }, path)


def run_and_compare(path: str,BLOCK_SIZE: int = 64):
    data = torch.load(path)
    gateup_output = data["gateup_output"].to("npu")
    reorder_topk_ids = data["reorder_topk_ids"].to("npu")
    scales = data["scales"].to("npu") if "scales" in data else None
    hidden_size = data["hidden_size"]
    start_expert_id = data["start_expert_id"]
    end_expert_id = data["end_expert_id"]

    down_input = torch.zeros_like(data["down_input"]).to("npu")

    # 重新计算输出
    gelu_and_mul_triton_launcher(
        gateup_output=gateup_output,
        down_input=down_input,
        reorder_topk_ids=reorder_topk_ids,
        scales=scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    expected_output = data["down_input"].to("npu")
    fail_ratio = check_accuracy(down_input, expected_output)


def run_and_save(path: str):
    data = torch.load(path, map_location="cpu")
    # # 升 fp32 计算结果
    # down_input = data["down_input"].to("npu").to(torch.float32)
    # gateup_output = data["gateup_output"].to("npu").to(torch.float32)
    # 原始 fp16 计算结果
    down_input = data["down_input"].to("npu")
    gateup_output = data["gateup_output"].to("npu")
    reorder_topk_ids = data["reorder_topk_ids"].to("npu")
    scales = data["w2_input_scale"]
    start_expert_id = data["start_expert_id"]
    end_expert_id = data["end_expert_id"]

    # 重新计算输出
    gelu_and_mul_triton_launcher(
        gateup_output=gateup_output,
        down_input=down_input,
        reorder_topk_ids=reorder_topk_ids,
        scales=scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        BLOCK_SIZE=128,
    )
    print(">> down_input:", down_input)
    print(">> down_input dtype:", down_input.dtype)
    # torch.save(down_input.cpu(), "OUTPUT_gelu_and_mul_triton_kernel_debug_npu_fp32.pt")
    torch.save(down_input.cpu(), "OUTPUT_gelu_and_mul_triton_kernel_debug_npu_fp16.pt")


if __name__ == "__main__":
    # 1. 对比模拟数据并检查精度
    # path = "gelu_mul_cuda_output.pt"
    # path = "gelu_mul_float_cuda_output.pt"
    # run_and_compare(path)
    # >>> Compare Type: float16
    # 精度达标 (0/512, 0.000000% <= 0.100000%)
    # >>> Compare Type: float32
    # Max diff at [5, 0]: test=1.2373121976852417, ref=1.2373123168945312, abs=1.1920928955078125e-07, rel=9.634527486923616e-08
    # 精度达标 (0/512, 0.000000% <= 0.010000%)


    # 2. 对比真实数据并检查精度
    # [REAL DATA INFO]
    # >> gateup_output:
    # Shape: torch.Size([8, 4096])
    # Dtype: torch.bfloat16
    # Device: cpu
    # First 10 elements: [0.9296875, 0.1162109375, 0.546875, -1.0859375, -0.10986328125, 0.5, 0.279296875, -0.0732421875, 0.1748046875, -0.4609375]
    # >> down_input:
    # Shape: torch.Size([8, 2048])
    # Dtype: torch.bfloat16
    # Device: cpu
    # First 10 elements: [-0.37890625, 0.369140625, -0.23046875, -0.003173828125, 0.041259765625, 0.287109375, 0.083984375, 0.0, 0.09521484375, -0.12109375]
    # >> reorder_topk_ids:
    # Shape: torch.Size([8])
    # Dtype: torch.int64
    # Device: cpu
    # First 10 elements: [19, 21, 103, 110, 184, 188, 240, 248]
    # >> w2_input_scale: None
    # >> start_expert_id: 160
    # >> end_expert_id: 191 
    key_mapping = {
        "gateup_output": "gateup_output",
        "down_input": "down_input",
        "reorder_topk_ids": "reorder_topk_ids",
        "scales": "w2_input_scale",
        "start_expert_id": "start_expert_id",
        "end_expert_id": "end_expert_id",
    }
    accuracy_dict = ["down_input"]
    src_path = "gelu_and_mul_triton_kernel_debug_cuda0.pt"
    expected_path = "gelu_and_mul_triton_kernel_expected_cuda0.pt"
    expected_output = torch.load("OUTPUT_gelu_and_mul_triton_kernel_debug_cuda0.pt", map_location="cpu")
    # 2.0 对比 expected_path 和 expected_output 的输出是否一致
    # expected_path_data = torch.load(expected_path, map_location="cpu")["down_input"]
    # check_accuracy(expected_path_data, expected_output)

    # 2.1 对比真实数据并检查精度
    # run_and_compare_real_data_npu(
    #     triton_kernel_impl=gelu_and_mul_triton_launcher,
    #     src_path=src_path,
    #     # expected_path=expected_path,
    #     expected_output=expected_output,
    #     key_mapping=key_mapping,
    #     accuracy=True,  # 是否检查精度
    #     accuracy_dict=accuracy_dict,
    #     USE_BLOCK_SIZE=True, # 使用自定义block_size, 非autotune情况下生效
    #     block_size=128,      # BLOCK_SIZE 设置
    # )
    # >>> Compare Type: bfloat16
    # Max diff at (tensor(4, device='npu:0'), tensor(129, device='npu:0')): test=-4.03125, ref=-4.0625, abs=0.03125, rel=0.007692305836826563
    # 精度不达标 (Mismatched elements:331/16384, 2.020264% > 0.000000%)

    # 3.0 测试 autotune kernel 的性能(真实数据)
    # run_and_compare_real_data_npu(
    #     triton_kernel_impl=gelu_and_mul_triton_launcher,
    #     src_path=src_path,
    #     key_mapping=key_mapping,
    #     accuracy=False,  # 是否检查精度
    #     autotune=True,  # 使用自动调优
    #     profiling=True,  # 进行性能分析
    # )
    # 4.0 三方精度对比
    # run_and_save(src_path)
    # NPU 升fp32 计算结果
    gold_tensor = torch.load("OUTPUT_gelu_and_mul_triton_kernel_debug_npu_fp32.pt", map_location="cpu")
    # NPU 原始fp16 计算结果
    act_tensor = torch.load("OUTPUT_gelu_and_mul_triton_kernel_debug_npu_fp16.pt", map_location="cpu")
    # GPU 原始fp16 计算结果
    std_tensor = torch.load("OUTPUT_gelu_and_mul_triton_kernel_debug_cuda0.pt", map_location="cpu")
    benchmark_compare_close(gold_tensor, act_tensor, std_tensor)
    # 测试结果如下:
    # act_re.max = 0.003834982169792056, std_re.max = 0.007187623996287584, limit ration = 10
    # act_re.sum = 5.553172588348389, std_re.sum = 7.714958190917969, limit ration = 2
    # act_small_error_ratio = 0.0, std_small_error_ratio = 0.0, limit ration = 2
    # act_rmse = 0.003285513259470463, std_rmse = 0.004117380827665329, limit ration = 2
