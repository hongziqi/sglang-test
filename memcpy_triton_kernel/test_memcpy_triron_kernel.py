import torch
import torch_npu
import triton
import triton.language as tl

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import check_accuracy, profiling_test_npu, run_and_compare_real_data_npu


# 定义自动调优配置
memcpy_triton_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 4096}),
        triton.Config(kwargs={"BLOCK_SIZE": 8192})
    ],
    key=[],
    # auto_profile_dir="/home/coder/.autotune/memcpy_triton_kernel",
)

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


memcpy_triton_kernel_autotuned = memcpy_triton_autotune(memcpy_triton_kernel)


def memcpy_triton_kernel_impl(
    dst_tensor: torch.Tensor,  # 目标指针 (num_tokens,)
    src_tensor: torch.Tensor,  # 源指针 (num_tokens,)
    offset_tensor: torch.Tensor,  # 偏移量指针 (1,)
    sz_tensor: torch.Tensor,  # 大小指针 (1,)
    offset_src: bool = False,  # 是否对源数据应用偏移
    chunk_size: int = 1,  # 块大小倍数
    BLOCK_SIZE: int = 256,  # 每个线程块的大小
    autotune: bool = False,  # 是否自动调优
):
    """
    执行内存复制操作。
    """
    max_size = min(dst_tensor.numel(), src_tensor.numel())
    if autotune:
        # print(">>> Using autotuned Triton kernel")
        # 使用自动调优的 Triton 内核
        grid = lambda meta: (triton.cdiv(max_size, meta["BLOCK_SIZE"]),)
        memcpy_triton_kernel_autotuned[grid](
            dst_ptr=dst_tensor,
            src_ptr=src_tensor,
            offset_ptr=offset_tensor,
            sz_ptr=sz_tensor,
            offset_src=offset_src,
            chunk_size=chunk_size,
        )
    else:
        # print(">>> Using regular Triton kernel, BLOCK_SIZE:", BLOCK_SIZE)
        # 使用普通的 Triton 内核
        grid = lambda meta: (triton.cdiv(max_size, BLOCK_SIZE),)
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
    src_tensor = torch.arange(num_tokens, dtype=torch.float32, device="npu")
    dst_tensor = torch.zeros_like(src_tensor, dtype=torch.float32, device="npu")
    offset_tensor = torch.tensor([offset], dtype=torch.int32, device="npu")
    size_tensor = torch.tensor([size], dtype=torch.int32, device="npu")
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


def run_and_compare(path):
    data = torch.load(path)
    src_tensor = data["src_tensor"].npu()
    dst_tensor = torch.zeros_like(src_tensor, dtype=torch.float32, device="npu")
    offset_tensor = data["offset_tensor"].npu()
    size_tensor = data["size_tensor"].npu()
    chunk_size = data["chunk_size"]
    BLOCK_SIZE = data["BLOCK_SIZE"]

    print("src_tensor: shape={}, dtype={}, device={}".format(
        src_tensor.shape, src_tensor.dtype, src_tensor.device))

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
    expected_output = data["dst_tensor"].npu()

    check_accuracy(dst_tensor, expected_output)


if __name__ == "__main__":
    # 1. 编译测试
    # path = "memcpy_npu_output.pt"
    # save_inputs_outputs(path)
    # Source Tensor:
    # [0.000e+00 1.000e+00 2.000e+00 ... 1.021e+03 1.022e+03 1.023e+03]

    # Destination Tensor (after memcpy):
    # [0.000e+00 1.000e+00 2.000e+00 ... 1.021e+03 1.022e+03 1.023e+03]

    # 2. 对比cuda和triton-ascend的输出
    # path = "memcpy_cuda_output.pt"
    # run_and_compare(path)
    # >>> Compare Type: float32
    # 精度达标 (0/1024, 0.000000% <= 0.010000%)

    # 3. 对比真实数据并检查精度
    key_mapping = {
        "dst_tensor": "dst",
        "src_tensor": "src",
        "offset_tensor": "offset",
        "sz_tensor": "sz",
        "offset_src": "offset_src",
        "chunk_size": "chunk_size",
        "BLOCK_SIZE": "BLOCK_SIZE",
    }
    accuracy_dict = ["dst_tensor"]
    src_path = "11_memcpy_triton_kernel_debug_cuda0.pt"
    expected_path = "11_memcpy_triton_kernel_expected_cuda0.pt"
    run_and_compare_real_data_npu(
        triton_kernel_impl=memcpy_triton_kernel_impl,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        accuracy=True,  # 是否检查精度
        accuracy_dict=accuracy_dict,
    )

    # 4.1 测试 autotune kernel 的性能 (BLOCK_SIZE:4096)
    run_and_compare_real_data_npu(
        triton_kernel_impl=memcpy_triton_kernel_impl,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        accuracy=False,  # 不检查精度
        autotune=True,  # 启用自动调优
        profiling=True,  # 启用性能分析
    )

    # 4.2 测试 normal kernel 的性能 (BLOCK_SIZE:8192)
    run_and_compare_real_data_npu(
        triton_kernel_impl=memcpy_triton_kernel_impl,
        src_path=src_path,
        expected_path=expected_path,
        key_mapping=key_mapping,
        accuracy=False,  # 不检查精度
        autotune=False,  # 不启用自动调优
        profiling=True,  # 启用性能分析
    )
