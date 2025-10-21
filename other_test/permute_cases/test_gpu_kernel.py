import pytest

import triton
import triton.language as tl

import torch
from typing import Optional


def foo(a, d ,shape ):
    y = a.reshape(shape)
    y = y.permute(0,2,1) + d
    return y


@triton.jit
def triton_gpu(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 4
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :] # (1, YBLOCK)
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None] # (XBLOCK, 1)
    xmask = xindex < xnumel
    x2 = xindex # (XBLOCK, 1)
    y3 = yindex # (1, YBLOCK)
    y0 = yindex % 2048  # (1, YBLOCK)
    y1 = (yindex // 2048) # (1, YBLOCK)
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last') # (XBLOCK, YBLOCK)
    tmp1 = tl.load(in_ptr1 + (y0 + (2048*x2) + (2048*4*y1)), xmask, eviction_policy='evict_last') # (XBLOCK, YBLOCK)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (4*y3)), tmp2, xmask)


def triton_foo_gpu(a, d, shape, dtype):
    # 仅用于 (Z=8, Y=2048, X=4) 的固定参数版本，以匹配 triton_gpu 的硬编码
    z, y, x = shape
    assert (z, y, x) == (8, 2048, 4), "triton_gpu 仅支持固定形状 (8, 2048, 4)"
    out = torch.empty_strided((z, x, y), (x * y, 1, x), device='npu', dtype=dtype)

    ynumel = y * z   # = 16384
    xnumel = x       # = 4
    XBLOCK = 4
    YBLOCK = 256
    assert ynumel % YBLOCK == 0

    grid = (xnumel // XBLOCK, ynumel // YBLOCK, 1)
    triton_gpu[grid](a, d, out, ynumel, xnumel, YBLOCK=YBLOCK, XBLOCK=XBLOCK)
    return out


def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None):
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


@pytest.mark.parametrize('dtype,sigtype', [(torch.float32, 'float32')])
@pytest.mark.parametrize('Z,Y,X', [(8, 2048, 4)])
def test_triton_gpu_kernel(Z, Y, X, dtype, sigtype):
    shape = (Z, Y, X)
    a = generate_tensor(shape=(Z, Y * X), dtype=sigtype).npu()
    d = generate_tensor(shape=(Z, X, Y), dtype=sigtype).npu()
    # 期望结果
    r_ref = foo(a, d, shape)
    # kernel 输出
    r = triton_foo_gpu(a, d, shape, dtype)
    validate_cmp(sigtype, r_ref, r)