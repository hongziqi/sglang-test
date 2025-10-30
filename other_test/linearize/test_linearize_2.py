import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu
import numpy as np


def torch_save_cache_to_buffer_with_mask(
    buffer,
    cache1,
    cache2,
    mask_int,
    buffer_stride,
    cache_stride,
    BLOCK,
    MASK_NUM
):
    idx = torch.arange(0, cache_stride)
    mask_idx = torch.arange(0, buffer_stride)
    mask = ((idx // BLOCK) % 2 == 0)
    max_len = min(buffer_stride, mask.shape[0])
    for i in range(buffer.shape[0]):
        if i % 2 == 0:
            tmp = cache1[i, 0, :]
            tmp[~(idx < MASK_NUM)] = 0
            tmp = tmp[mask]
            buffer[i, 0, :max_len] = tmp[:max_len]
        else:
            # tmp = cache2[i, 0, mask]
            # tmp[~((mask_idx < MASK_NUM) & (mask_idx < mask_int[i]))] = 0
            # buffer[i, 0, :max_len] = tmp[:max_len]
            tmp = cache2[i, 0, :]
            tmp[~(idx < MASK_NUM)] = 0
            tmp = tmp[mask]
            buffer[i, 0, :max_len] = tmp[:max_len]


@triton.jit
def save_cache_to_buffer_with_mask(
    buffer_ptr,
    cache_ptr1,
    cache_ptr2,
    mask_int_ptr,
    dbg,
    dbg2,
    buffer_stride: tl.constexpr,
    BLOCK: tl.constexpr,
    MASK_NUM: tl.constexpr
):
    pid_loc = tl.program_id(0)

    buffer_offset = pid_loc * buffer_stride

    buffer_index = tl.arange(0, buffer_stride)
    index = buffer_offset + buffer_index
    cache_index_0 = index // BLOCK
    cache_index_1 = index % BLOCK
    # mask_int = tl.load(mask_int_ptr + pid_loc)
    mask = (2*BLOCK*cache_index_0 + cache_index_1) < MASK_NUM
    
    mask_dbg = tl.tensor(mask, tl.int32)
    a = tl.full(mask_dbg.shape, 1, tl.int32)
    b = tl.full(mask_dbg.shape, 0, tl.int32)
    tm = tl.where(mask, a, b)
    tl.store(dbg + index, tm)

    for i in range(0, buffer_stride):
        idx = buffer_offset + i
        tl.store(dbg2 + idx, 2*BLOCK*(idx // BLOCK) + (idx % BLOCK))

    if pid_loc % 2 == 0:
        tmp = tl.load(cache_ptr1 + (2*BLOCK*cache_index_0 + cache_index_1), \
            mask)
        tl.store(buffer_ptr + index, tmp)
    if pid_loc % 2 == 1:
        tmp = tl.load(cache_ptr2 + (2*BLOCK*cache_index_0 + cache_index_1), \
            mask)
        tl.store(buffer_ptr + index, tmp)


def biggest_divisor(num):
    for i in range(2, num):  
        if num % i == 0:  
            return num // i
    return num


types = [
    # (torch.float32, 'float32'),
    # (torch.float16, 'float16'),
    # (torch.bfloat16, 'bfloat16'),
    (torch.int8, 'int8'),
    # (torch.int16, 'int16'),
    # (torch.int32, 'int32'),
    # (torch.int64, 'int64'),
]

shapes = [
    (5, 15),
    # (16, 53),
    # (2048, 16),
    # (512, 25),
    # (32, 35),
    # (128, 14),
]

def write_tensor_full_text(path: str, tensor: torch.Tensor, name="default", mode="a"):
    t = tensor.detach().cpu().numpy()
    s = np.array2string(t, separator=', ')
    meta = f"# Tensor {name}, shape: {tensor.shape}, dtype: {tensor.dtype}\n"
    with open(path, mode) as f:
        f.write(meta + "\n")
        f.write(s)
        f.write("\n\n")

@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', shapes)
def test_linearize_jump_load_with_mask(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    cache_len = buffer_len * 2
    buffer_ref = torch.zeros(batch_size, 1, buffer_len, dtype=dtype)
    buffer = buffer_ref.npu()
    cache1_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache1 = cache1_ref.npu()
    cache2_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache2 = cache2_ref.npu()
    mask_ref = torch.arange(0, batch_size, dtype=torch.int64)*2
    mask = mask_ref.npu()
    mask_num = batch_size * 1 * cache_len

    dbg = torch.zeros(batch_size, 1, buffer_len, dtype=dtype).npu()
    dbg2 = torch.zeros(batch_size, 1, buffer_len, dtype=dtype).npu()
    print(f"block={block}, buffer_len={buffer_len}, cache_len={cache_len}, mask_num={mask_num}")
    print(f"cache1.shape={cache1.shape}, cache2.shape={cache2.shape}, buffer.shape={buffer.shape}")
    torch_save_cache_to_buffer_with_mask(buffer_ref, cache1_ref, cache2_ref, mask_ref, buffer_len, cache_len, block, mask_num)
    save_cache_to_buffer_with_mask[(batch_size, 1, 1)](buffer, cache1, cache2, mask, dbg, dbg2, buffer_len, block, mask_num)
    print("dbg:", dbg)
    print("dbg2:", dbg2)
    test_common.validate_cmp(sigtype, buffer, buffer_ref)
