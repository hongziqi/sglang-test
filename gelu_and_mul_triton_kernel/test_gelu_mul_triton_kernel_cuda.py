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


def save_inputs_outputs(path: str, token_num: int = 8, hidden_size: int = 128,
                        start_expert_id=0, end_expert_id=31, BLOCK_SIZE: int = 64):
    gateup_output = torch.ones((token_num, hidden_size), device="cuda", dtype=torch.float32)
    down_input = torch.empty(token_num, hidden_size // 2, device="cuda", dtype=torch.float32)

    reorder_topk_ids = torch.randint(
        low=start_expert_id,
        high=end_expert_id + 1,
        size=(token_num,),
        device="cuda",
        dtype=torch.int32,
    )

    scales = torch.rand(end_expert_id - start_expert_id + 1, device="cuda", dtype=torch.float32)

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
        "gateup_output": gateup_output.cpu(),
        "down_input": down_input.cpu(),
        "reorder_topk_ids": reorder_topk_ids.cpu(),
        "scales": scales.cpu(),
        "start_expert_id": start_expert_id,
        "end_expert_id": end_expert_id,
        "hidden_size": hidden_size,
    }, path)

def run_and_compare(path: str, atol: float = 1e-3, rtol: float = 1e-3):
    data = torch.load(path)
    gateup_output = data["gateup_output"].cuda()
    reorder_topk_ids = data["reorder_topk_ids"].cuda()
    scales = data["scales"].cuda()
    start_expert_id = data["start_expert_id"]
    end_expert_id = data["end_expert_id"]
    hidden_size = data["hidden_size"]

    down_input = torch.zeros_like(data["down_input"]).cuda()

    # 重新计算输出
    gelu_and_mul_triton_launcher(
        gateup_output=gateup_output,
        down_input=down_input,
        reorder_topk_ids=reorder_topk_ids,
        scales=scales,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        hidden_size=hidden_size,
    )

    output_ref = data["down_input"].cuda()
    is_close = torch.isclose(down_input, output_ref, rtol=rtol, atol=atol)
    mismatch_idx = torch.nonzero(~is_close)
    print(f"Output consistent: {is_close.all().item()}\nMax difference: {(down_input - output_ref).abs().max().item()}")
    for idx in mismatch_idx:
        i, j = idx.tolist()
        print(f"[{i}, {j}]: test={down_input[i, j]}, ref={output_ref[i, j]}, diff={abs(down_input[i, j] - output_ref[i, j])}")

   
if __name__ == "__main__":
    path = "gelu_mul_cuda_output.pt"
    save_inputs_outputs(path)
    run_and_compare(path)
