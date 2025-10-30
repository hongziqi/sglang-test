为什么 mask 中会存在 0 的情况，理想应该是全1 （偶现）

platform linux -- Python 3.11.14, pytest-8.3.2, pluggy-1.6.0 -- /home/coder/miniconda/envs/triton/bin/python3.11
cachedir: .pytest_cache
rootdir: /home/coder/workspace/triton-ascend
configfile: pyproject.toml
plugins: xdist-3.6.1
collected 1 item                                                                                                                                                                                                                   

ascend/examples/pytest_ut/test_linearize_2.py::test_linearize_jump_load_with_mask[5-15-dtype0-int8] block=5, buffer_len=15, cache_len=30, mask_num=150
cache1.shape=torch.Size([5, 1, 30]), cache2.shape=torch.Size([5, 1, 30]), buffer.shape=torch.Size([5, 1, 15])
Dumping launcher_cxx11abi1.cxx to /home/coder/.triton/dump/RNZJwMqVqQjswa2RwGBDizSwD2aCpTCZiA0nGpGl_nY
dbg: tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],

        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],

        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],

        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]], device='npu:0',
       dtype=torch.int8)
dbg2: tensor([[[   0,    1,    2,    3,    4,   10,   11,   12,   13,   14,   20,
            21,   22,   23,   24]],

        [[  30,   31,   32,   33,   34,   40,   41,   42,   43,   44,   50,
            51,   52,   53,   54]],

        [[  60,   61,   62,   63,   64,   70,   71,   72,   73,   74,   80,
            81,   82,   83,   84]],

        [[  90,   91,   92,   93,   94,  100,  101,  102,  103,  104,  110,
           111,  112,  113,  114]],

        [[ 120,  121,  122,  123,  124, -126, -125, -124, -123, -122, -116,
          -115, -114, -113, -112]]], device='npu:0', dtype=torch.int8)
FAILED

============================================================================================================= FAILURES =============================================================================================================
_______________________________________________________________________________________ test_linearize_jump_load_with_mask[5-15-dtype0-int8] _______________________________________________________________________________________

batch_size = 5, buffer_len = 15, dtype = torch.int8, sigtype = 'int8'

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
>       test_common.validate_cmp(sigtype, buffer, buffer_ref)

ascend/examples/pytest_ut/test_linearize_2.py:143: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

dtype = 'int8'
y_cal = tensor([[[ 99,  56,  55,  96,  13,  74, 103,  94,  67,   7,  65,  16,   0,   0,
            0]],

        [[ 62, 117, ...9,  89,  38,   6,  49,  88, 115, 116,  85,  76,   8,  87,   0,   0,
            0]]], device='npu:0', dtype=torch.int8)
y_ref = tensor([[[ 99,  56,  55,  96,  13,  74, 103,  94,  67,   7,  65,  16,  57,  93,
           22]],

        [[ 62, 117, ...9,  89,  38,   6,  49,  88, 115, 116,  85,  76,   8,  87, 120,  65,
          121]]], device='npu:0', dtype=torch.int8)
overflow_mode = None

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
>           assert torch.equal(y_cal, y_ref)
E           AssertionError: assert False
E            +  where False = <built-in method equal of type object at 0xffff86678460>(tensor([[[ 99,  56,  55,  96,  13,  74, 103,  94,  67,   7,  65,  16,   0,   0,\n            0]],\n\n        [[ 62, 117,  35,  17,  66,  43, 106,  24,  46,   9,  87,   1,   0,   0,\n            0]],\n\n        [[ 72, 104,   7,  78,  34,   7,  12,  59,  29,  15, 103, 120, 125,  28,\n           46]],\n\n        [[ 86,  38,  23, 109,  64,  72, 102, 102,  13,  72,  51,  60,  17,  84,\n           35]],\n\n        [[ 79,  89,  38,   6,  49,  88, 115, 116,  85,  76,   8,  87,   0,   0,\n            0]]], device='npu:0', dtype=torch.int8), tensor([[[ 99,  56,  55,  96,  13,  74, 103,  94,  67,   7,  65,  16,  57,  93,\n           22]],\n\n        [[ 62, 117,  35,  17,  66,  43, 106,  24,  46,   9,  87,   1,   1, 100,\n            1]],\n\n        [[ 72, 104,   7,  78,  34,   7,  12,  59,  29,  15, 103, 120, 125,  28,\n           46]],\n\n        [[ 86,  38,  23, 109,  64,  72, 102, 102,  13,  72,  51,  60,  17,  84,\n           35]],\n\n        [[ 79,  89,  38,   6,  49,  88, 115, 116,  85,  76,   8,  87, 120,  65,\n          121]]], device='npu:0', dtype=torch.int8))
E            +    where <built-in method equal of type object at 0xffff86678460> = torch.equal

ascend/examples/pytest_ut/test_common.py:128: AssertionError
===================================================================================================== short test summary info ======================================================================================================
FAILED ascend/examples/pytest_ut/test_linearize_2.py::test_linearize_jump_load_with_mask[5-15-dtype0-int8] - AssertionError: assert False
======================================================================================================== 1 failed in 24.00s ========================================================================================================