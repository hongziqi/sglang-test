import torch
#import torch_npu
import triton
import triton.language as tl

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

def gelu_and_mul_triton_launcher(
    gateup_output: torch.Tensor,     # shape: [token_num, hidden_size]
    down_input: torch.Tensor,        # shape: [token_num, hidden_size // 2]
    reorder_topk_ids: torch.Tensor,  # shape: [token_num], 每个 token 对应的 expert id
    scales: torch.Tensor | None,     # shape: [expert_range] 缩放因子
    hidden_size: int,
    start_expert_id: int,
    end_expert_id: int,
    BLOCK_SIZE: int = 64,
):
    grid = (reorder_topk_ids.shape[0],)  # 每个 token 一个 program
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
    gateup_output = torch.ones((token_num, hidden_size), device="npu", dtype=torch.float16)
    down_input = torch.empty(token_num, hidden_size // 2, device="npu", dtype=torch.float16)

    # 模拟每个 token 对应的 expert id（范围在 start_expert_id 到 end_expert_id 之间）
    reorder_topk_ids = torch.randint(
        low=start_expert_id,
        high=end_expert_id + 1,
        size=(token_num,),
        device="npu",
        dtype=torch.int32,
    )

    # 可选：缩放因子 scales，None 表示不使用 scale
    scales = torch.rand(end_expert_id - start_expert_id + 1, device="npu", dtype=torch.float16)

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


def check_accuracy(output: torch.Tensor, expected: torch.Tensor):
    # 根据 dtype 自动判定阈值
    dtype = expected.dtype
    if dtype == torch.float16:
        print(">>> Compare Type: float16")
        rtol, atol, max_fail_ratio = 1e-3, 1e-3, 1e-3  # 双千分之一
    elif dtype == torch.float32:
        print(">>> Compare Type: float32")
        rtol, atol, max_fail_ratio = 1e-4, 1e-4, 1e-4  # 双万分之一
    elif dtype in [torch.int8, torch.uint8]:
        print(">>> Compare Type: int8 | uint8")
        rtol, atol, max_fail_ratio = 1e-3, 1, 1e-3     # 容差为1
    else:
        raise ValueError(f"Unsupported dtype for accuracy check: {dtype}")

    # 计算误差
    abs_diff = (output - expected).abs()
    rel_diff = abs_diff / (expected.abs() + 1e-6)
    fail_mask = (abs_diff > atol) & (rel_diff > rtol)

    total = output.numel()
    fail = fail_mask.sum().item()
    fail_ratio = fail / total

    # 打印最大误差点
    max_abs = abs_diff.max().item()
    if max_abs > 0:
        max_idx_flat = torch.argmax(abs_diff).item()
        i, j = divmod(max_idx_flat, output.shape[1])
        print(f"Max diff at [{i}, {j}]: test={output[i, j].item()}, "
            f"ref={expected[i, j].item()}, "
            f"abs={abs_diff[i, j].item()}, rel={rel_diff[i, j].item()}")

    # 判断是否精度达标
    if fail_ratio <= max_fail_ratio:
        print(f"精度达标 ({fail}/{total}, {fail_ratio:.6%} <= {max_fail_ratio:.6%})")
    else:
        print(f"精度不达标 ({fail}/{total}, {fail_ratio:.6%} > {max_fail_ratio:.6%})")
        idx_list = torch.nonzero(fail_mask)[:10]
        for i, j in idx_list.tolist():
            print(f"[{i},{j}]: test={output[i, j].item():.6f}, "
                  f"ref={expected[i, j].item():.6f}, "
                  f"diff={abs_diff[i, j].item():.6f}, rel={rel_diff[i, j].item():.6f}")

    return fail_ratio


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


if __name__ == "__main__":
    # path = "gelu_mul_cuda_output.pt"
    path = "gelu_mul_float_cuda_output.pt"
    run_and_compare(path)
    # >>> Compare Type: float16
    # 精度达标 (0/512, 0.000000% <= 0.100000%)
    # >>> Compare Type: float32
    # Max diff at [5, 0]: test=1.2373121976852417, ref=1.2373123168945312, abs=1.1920928955078125e-07, rel=9.634527486923616e-08
    # 精度达标 (0/512, 0.000000% <= 0.010000%)
