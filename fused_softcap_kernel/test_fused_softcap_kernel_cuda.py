import torch
import triton
import triton.language as tl


# 针对 fused_softcap_kernel 的自动调优配置（triton.autotune 是 Triton 的一个特性，用于自动选择最优的配置）
# 在运行时，依据n_ele的大小自动选择最优的BLOCK_SIZE和num_warps
fused_softcap_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 32768}, num_warps=32),
    ],
    key=["n_ele"],
)


@triton.jit
def fused_softcap_kernel(
    output_ptr,
    input_ptr,
    n_ele,
    softcap_const: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele
    x = tl.load(input_ptr + offsets, mask=mask)
    fx = x.to(tl.float32)
    fxs = fx / softcap_const
    exped = tl.exp(2 * fxs) 
    top = exped - 1
    bottom = exped + 1
    # softcap(x;τ) = τ * tanh(x / τ), 这里tanh(z) 用 exp(2z) - 1 / exp(2z) + 1 代替
    # τ 是 softcap_const
    # z = x / τ, 即 z 为 fxs
    output = top / bottom * softcap_const 
    tl.store(output_ptr + offsets, output, mask=mask)


fused_softcap_kernel_autotuned = fused_softcap_autotune(fused_softcap_kernel)

def fused_softcap_impl(x, softcap_const, autotune=False):
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    n_elements = output.numel()
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        fused_softcap_kernel_autotuned[grid](output, x, n_elements, softcap_const)
    else:
        fused_softcap_kernel[(triton.cdiv(n_elements, 128),)](
            output, x, n_elements, softcap_const, BLOCK_SIZE=128, num_warps=8
        )
    return output


def save_inputs_outputs(path, n_elements=1024, softcap_const=0.5):
    # 输入规格
    # x (n_elements, ) float32
    # softcap_const float32
    input_tensor = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    output_tensor = fused_softcap_impl(input_tensor, softcap_const)

    # 输出数据
    # print("input_tensor:", input_tensor)
    # print("output_tensor:", output_tensor)

    torch.save({
        'input_tensor': input_tensor.cpu(),
        'output_tensor': output_tensor.cpu(),
        'softcap_const': softcap_const,
    }, path)


def run_and_compare(path, rtol=1e-3, atol=1e-3):
    data = torch.load(path)
    input_tensor = data['input_tensor'].cuda()
    softcap_const = data['softcap_const']
    
    output = fused_softcap_impl(input_tensor, softcap_const)

    expected_output = data['output_tensor'].cuda()

    assert torch.allclose(output, expected_output, rtol=rtol, atol=atol), \
        f"Output mismatch: {output} vs {expected_output}"
    
    print("Output consistent: True")


if __name__ == "__main__":
    path = "fused_softcap_kernel_cuda_output.pt"
    save_inputs_outputs(path)

    run_and_compare(path)
    # Output consistent: True
