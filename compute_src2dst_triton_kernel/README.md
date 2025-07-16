(triton) (triton) coder@candy-npu:~/workspace/sglang-test/compute_src2dst_triton_kernel$ npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0.3                 Version: 24.1.0.3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 6     910B4               | OK            | 85.6        38                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          2856 / 32768         |
+===========================+===============+====================================================+
| 7     910B4               | OK            | 88.7        39                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          2840 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 6                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
(triton) (triton) coder@candy-npu:~/workspace/sglang-test/compute_src2dst_triton_kernel$ python test_compute_src2dst_triton_kernel.py 
>> reorder_ids: [0 1 2 3 4 5]
>> Block Size: 128
>> Compute src2dst: [0 0 1 2 3 4]
>> Expected output: [0 1 2 3 4 5]
>>> Compare Type: int32
Max diff at (tensor(1, device='npu:0'),): test=0, ref=1, abs=1, rel=0.9999990463256836
精度不达标 (5/6, 83.333333% > 0.100000%)
(1,): test=0.000000, ref=1.000000, diff=1.000000, rel=0.999999
(2,): test=1.000000, ref=2.000000, diff=1.000000, rel=0.500000
(3,): test=2.000000, ref=3.000000, diff=1.000000, rel=0.333333
(4,): test=3.000000, ref=4.000000, diff=1.000000, rel=0.250000
(5,): test=4.000000, ref=5.000000, diff=1.000000, rel=0.200000
reorder_ids: tensor([0, 1, 2, 3, 4, 5], device='npu:0', dtype=torch.int32)
src2dst: tensor([0, 0, 1, 2, 3, 4], device='npu:0', dtype=torch.int32)
[W716 03:39:22.793892537 compiler_depend.ts:26] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
Traceback (most recent call last):
  File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 132, in <module>
    test_compute_src2dst_triton_no_conflict()
  File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 107, in test_compute_src2dst_triton_no_conflict
    assert np.array_equal(excepted, actual), f"Expected {excepted}, but got {actual}"
AssertionError: Expected [0 1 2 3 4 5], but got [0 0 1 2 3 4]
[ERROR] 2025-07-16-03:39:23 (PID:67079, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
(triton) (triton) coder@candy-npu:~/workspace/sglang-test/compute_src2dst_triton_kernel$ TRITON_INTERPRET=1 python test_compute_src2dst_triton_kernel.py 
>> reorder_ids: [0 1 2 3 4 5]
>> Block Size: 128
>> Compute src2dst: [0 1 2 3 4 5]
>> Expected output: [0 1 2 3 4 5]
>>> Compare Type: int32
精度达标 (0/6, 0.000000% <= 0.100000%)
reorder_ids: tensor([0, 1, 2, 3, 4, 5], device='npu:0', dtype=torch.int32)
src2dst: tensor([0, 1, 2, 3, 4, 5], device='npu:0', dtype=torch.int32)
[W716 03:40:53.101599275 compiler_depend.ts:26] Warning: Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu.If you are more concerned about high-performance execution,please cast dtype to float32. (function operator())
Test passed!
