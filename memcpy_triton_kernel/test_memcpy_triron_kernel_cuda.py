import torch
import triton
import triton.language as tl


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy


# 定义 memcpy_triton_kernel
@triton.jit
def memcpy_triton_kernel(
    dst_ptr,    # 目标张量指针
    src_ptr,    # 源张量指针
    offset_ptr, # 偏移（拷贝的起始位置）
    sz_ptr,     # 要拷贝的数据长度
    offset_src, # 是否对源进行偏移
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,   # 每个线程块处理的元素数
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def memcpy_triton_kernel_impl(
    dst_tensor: torch.Tensor,  # 目标指针 (num_tokens,)
    src_tensor: torch.Tensor,  # 源指针 (num_tokens,)
    offset_tensor: torch.Tensor,  # 偏移量指针 (1,)
    sz_tensor: torch.Tensor,  # 大小指针 (1,)
    offset_src: bool = False,  # 是否对源数据应用偏移
    chunk_size: int = 1,  # 块大小倍数
    BLOCK_SIZE: int = 256,  # 每个线程块的大小
):
    """
    执行内存复制操作。
    """
    max_size = min(dst_tensor.numel(), src_tensor.numel())
    grid = lambda meta: (triton.cdiv(max_size, BLOCK_SIZE),)

    # 启动 Triton 内核
    memcpy_triton_kernel[grid](
        dst_ptr=dst_tensor,
        src_ptr=src_tensor,
        offset_ptr=offset_tensor,
        sz_ptr=sz_tensor,
        offset_src=offset_src,
        chunk_size=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def save_inputs_outputs(
        path: str,
        num_tokens: int = 1024,
        offset: int = 0,
        size: int = 1024,
        chunk_size: int = 1,
        BLOCK_SIZE: int = 256
):
    # 构造输入数据
    src_tensor = torch.arange(num_tokens, dtype=torch.float32, device="cuda")
    dst_tensor = torch.zeros_like(src_tensor, dtype=torch.float32, device="cuda")
    offset_tensor = torch.tensor([offset], dtype=torch.int32, device="cuda")
    size_tensor = torch.tensor([size], dtype=torch.int32, device="cuda")
    # 执行 Triton 内核
    memcpy_triton_kernel_impl(
        dst_tensor=dst_tensor,
        src_tensor=src_tensor,
        offset_tensor=offset_tensor,
        sz_tensor=size_tensor,
        offset_src=False,  # 不对源数据应用偏移
        chunk_size=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # 输出结果
    print("Source Tensor:")
    print(src_tensor.cpu().numpy())
    print("\nDestination Tensor (after memcpy):")
    print(dst_tensor.cpu().numpy())

    # 保存输入输出
    torch.save({
        "src_tensor": src_tensor.cpu(),
        "dst_tensor": dst_tensor.cpu(),
        "offset_tensor": offset_tensor.cpu(),
        "size_tensor": size_tensor.cpu(),
        "chunk_size": chunk_size,
        "BLOCK_SIZE": BLOCK_SIZE,
    }, path)


def run_and_compare(path, atol: float = 1e-3, rtol: float = 1e-3):
    data = torch.load(path)
    src_tensor = data["src_tensor"].cuda()
    dst_tensor = torch.zeros_like(src_tensor, dtype=torch.float32, device="cuda")
    offset_tensor = data["offset_tensor"].cuda()
    size_tensor = data["size_tensor"].cuda()
    chunk_size = data["chunk_size"]
    BLOCK_SIZE = data["BLOCK_SIZE"]

    # 重新计算输出
    memcpy_triton_kernel_impl(
        dst_tensor=dst_tensor,
        src_tensor=src_tensor,
        offset_tensor=offset_tensor,
        sz_tensor=size_tensor,
        offset_src=False,  # 不对源数据应用偏移
        chunk_size=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # 检查结果
    output_ref = data["dst_tensor"].cuda()
    is_close = torch.isclose(dst_tensor, output_ref, atol=atol, rtol=rtol)
    mismatch_idx = torch.nonzero(~is_close)
    print(f"Output consistent: {is_close.all().item()}\nMax difference: {(dst_tensor - output_ref).abs().max().item()}")
    for idx in mismatch_idx:
        i, j = idx.tolist()
        print(f"[{i}, {j}]: test={dst_tensor[i, j]}, ref={output_ref[i, j]}, diff={abs(dst_tensor[i, j] - output_ref[i, j])}")


def run_and_compare_real_data(src_path, expected_path):
    """
    [MEMCPY TRITON KERNEL REAL DATA]
    dst:
    Shape: torch.Size([4, 2048])
    Dtype: torch.bfloat16
    Device: cpu
    First 5 elements: [0.0, 0.0, 0.0, 0.0, 0.0]
    src:
    Shape: torch.Size([4, 2048])
    Dtype: torch.bfloat16
    Device: cpu
    First 5 elements: [0.0045166015625, -0.00823974609375, 0.0179443359375, 0.01300048828125, 3.0517578125e-05]
    offset:
    Shape: torch.Size([])
    Dtype: torch.int64
    Device: cpu
    First 5 elements: [0]
    sz:
    Shape: torch.Size([])
    Dtype: torch.int32
    Device: cpu
    First 5 elements: [2]
    offset_src: False
    chunk_size: 2048
    BLOCK_SIZE: 8192
    """
    try:
        data = torch.load(src_path)
    except FileNotFoundError:
        print(f"File {src_path} not found. Please run the test to generate it.")
        return
    print("\n[MEMCPY TRITON KERNEL REAL DATA]")

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {value.cpu().shape}")
            print(f"  Dtype: {value.cpu().dtype}")
            print(f"  Device: {value.cpu().device}")
            # 打印前5个元素
            print(f"  First 10 elements: {value.cpu().flatten()[:10].tolist()}")
        else:
            print(f"{key}: {value}")
    
    src_tensor = data["src"].cuda()
    dst_tensor = data["dst"].cuda()
    offset_tensor = data["offset"].cuda()
    size_tensor = data["sz"].cuda()
    offset_src = data["offset_src"]
    chunk_size = data["chunk_size"]
    BLOCK_SIZE = data["BLOCK_SIZE"]

    # 重新计算输出
    memcpy_triton_kernel_impl(
        dst_tensor=dst_tensor,
        src_tensor=src_tensor,
        offset_tensor=offset_tensor,
        sz_tensor=size_tensor,
        offset_src=offset_src,  # 是否对源数据应用偏移
        chunk_size=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 存储新的路径
    torch.save({
        "src": src_tensor.cpu(),
        "dst": dst_tensor.cpu(),
        "offset": offset_tensor.cpu(),
        "sz": size_tensor.cpu(),
        "offset_src": offset_src,
        "chunk_size": chunk_size,
        "BLOCK_SIZE": BLOCK_SIZE,
    }, expected_path)


# fffrog testcases
def run_memcpy_kernel():
    # 定义输入和输出张量
    device = torch.device("npu")
    src_tensor = torch.arange(1024, dtype=torch.int32, device=device)  # 源张量
    dst_tensor = torch.zeros_like(src_tensor)  # 目标张量

    # 定义偏移量和大小
    offset_tensor = torch.tensor([0], dtype=torch.int32, device=device)  # 偏移量
    size_tensor = torch.tensor([1024], dtype=torch.int32, device=device)  # 数据大小

    # 配置参数
    BLOCK_SIZE = 256  # 每个线程块处理的数据大小
    grid_size = triton.cdiv(1024, BLOCK_SIZE)  # 计算网格大小

    # 调用 Triton 内核
    memcpy_triton_kernel[(grid_size,)](
        dst_tensor,  # 目标指针
        src_tensor,  # 源指针
        offset_tensor,  # 偏移量指针
        size_tensor,  # 大小指针
        offset_src=False,  # 是否对源数据应用偏移
        chunk_size=1,  # 块大小倍数
        BLOCK_SIZE=BLOCK_SIZE,  # 每个线程块的大小
    )

    # 打印结果
    print("Source Tensor:")
    print(src_tensor.cpu().numpy())
    print("\nDestination Tensor (after memcpy):")
    print(dst_tensor.cpu().numpy())


if __name__ == "__main__":
    # 1. 运行并比较结果
    # path = "memcpy_cuda_output.pt"
    # save_inputs_outputs(path, num_tokens=1024, offset=1, size=1023, chunk_size=1, BLOCK_SIZE=256)
    # run_and_compare(path)
    # Source Tensor:
    # [0.000e+00 1.000e+00 2.000e+00 ... 1.021e+03 1.022e+03 1.023e+03]

    # Destination Tensor (after memcpy):
    # [0.000e+00 0.000e+00 1.000e+00 ... 1.020e+03 1.021e+03 1.022e+03]
    # Output consistent: True
    # Max difference: 0.0

    # 2. 运行真实数据, 并保存运行结果
    src_path = "11_memcpy_triton_kernel_debug_cuda0.pt"
    expected_path = "11_memcpy_triton_kernel_expected_cuda0.pt"
    run_and_compare_real_data(src_path, expected_path)
