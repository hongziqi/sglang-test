—————————————————————————————————————————————————————————— 编译通过时，精度不达标
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

—————————————————————————————————————————————————————————— 编译偶发 507053 报错
>> reorder_ids: [0 1 2 3 4 5]
>> Block Size: 128
>> Compute src2dst: [0 1 2 3 4 5]
>> Expected output: [0 1 2 3 4 5]
>>> Compare Type: int32
精度达标 (0/6, 0.000000% <= 0.100000%)
reorder_ids: Traceback (most recent call last):
  File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 132, in <module>
    test_compute_src2dst_triton_no_conflict()
  File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 101, in test_compute_src2dst_triton_no_conflict
    print("reorder_ids:", render_ids)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/torch/_tensor.py", line 568, in __repr__
    return torch._tensor_str._str(self, tensor_contents=tensor_contents)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/torch/_tensor_str.py", line 704, in _str
    return _str_intern(self, tensor_contents=tensor_contents)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/torch/_tensor_str.py", line 621, in _str_intern
    tensor_str = _tensor_str(self, indent)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/torch/_tensor_str.py", line 353, in _tensor_str
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/torch/_tensor_str.py", line 141, in __init__
    value_str = f"{value}"
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/torch/_tensor.py", line 1097, in __format__
    return self.item().__format__(format_spec)
RuntimeError: operator():build/CMakeFiles/torch_npu.dir/compiler_depend.ts:26 NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507035
[ERROR] 2025-07-22-08:09:53 (PID:28849, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EZ9999: Inner Error!
EZ9999: [PID: 28849] 2025-07-22-08:09:52.219.466 The error from device(chipId:4, dieId:0), serial number is 1, there is an exception of aivec error, core id is 49, error code = 0x800000, dump info: pc start: 0x124000000000, current: 0x124000000254, vec error info: 0xf90e756ec1, mte error info: 0x6000097, ifu error info: 0x2ad7ef7522480, ccu error info: 0x867d7abf787d4e8c, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c102804800.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_core_proc.cc][LINE:293]
        TraceBack (most recent call last):
       The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x6000097, fixp_error1 info: 0, fsmId:0, tslot:0, thread:0, ctxid:0, blk:0, sublk:0, subErrType:4.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_core_proc.cc][LINE:312]
       Kernel task happen error, retCode=0x31, [vector core exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1539]
       AIV Kernel happen error, retCode=0x31.[FUNC:GetError][FILE:stream.cc][LINE:1190]
       [AIC_INFO] after execute:args print end[FUNC:GetError][FILE:stream.cc][LINE:1190]
       Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=18, flip_num=0, fault kernel_name=compute_src2dst_triton_kernel_0, fault kernel info ext=compute_src2dst_triton_kernel, program id=0, hash=15011427645934493387.[FUNC:GetError][FILE:stream.cc][LINE:1190]
       rtStreamSynchronizeWithTimeout execute failed, reason=[vector core exception][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
       synchronize stream failed, runtime result = 507035[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]

[W722 08:09:53.771756236 compiler_depend.ts:526] Warning: NPU warning, error code is 507035[Error]: 
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[vector core exception][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 28849] 2025-07-22-08:09:53.658.509 wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function npuSynchronizeUsedDevices)
[W722 08:09:53.773865436 compiler_depend.ts:508] Warning: NPU warning, error code is 507035[Error]: 
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[vector core exception][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 28849] 2025-07-22-08:09:53.661.032 wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function npuSynchronizeDevice)
[W722 08:09:53.775671005 compiler_depend.ts:151] Warning: NPU warning, error code is 507035[Error]: 
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[vector core exception][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 28849] 2025-07-22-08:09:53.662.571 wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function empty_cache)
[W722 08:09:53.777502883 compiler_depend.ts:508] Warning: NPU warning, error code is 507035[Error]: 
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[vector core exception][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 28849] 2025-07-22-08:09:53.664.727 wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function npuSynchronizeDevice)
[W722 08:09:53.778778100 compiler_depend.ts:151] Warning: NPU warning, error code is 507035[Error]: 
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[vector core exception][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 28849] 2025-07-22-08:09:53.666.242 wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function empty_cache)


—————————————————————————————————————————————————————————— NPU 驱动太低会导致507057 报错 （与507053 报错信息初步判断一致）
(triton) coder@candy-npu:~/workspace/sglang-test/compute_src2dst_triton_kernel$ npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.6                   Version: 23.0.6                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 4     910B4               | OK            | 89.9        38                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          2808 / 32768         |
+===========================+===============+====================================================+
| 7     910B4               | OK            | 88.8        40                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          2807 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 4                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
(triton) coder@candy-npu:~/workspace/sglang-test/compute_src2dst_triton_kernel$ python test_compute_src2dst_triton_kernel.py 
[W722 01:57:23.467290876 compiler_depend.ts:57] Warning: EZ9999: Inner Error!
EZ9999: [PID: 18805] 2025-07-22-01:57:22.025.595 The error from device(chipId:4, dieId:0), serial number is 2, there is an exception of aivec error, core id is 32, error code = 0x800000, dump info: pc start: 0x124041000000, current: 0x124041000254, vec error info: 0x32051105a9, mte error info: 0x6000097, ifu error info: 0x7ae3f3701ba40, ccu error info: 0xf6708a1870080c0f, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd000288, para base: 0x124102800400.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_core_proc.cc][LINE:293]
        TraceBack (most recent call last):
       The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x6000097, fixp_error1 info: 0, fsmId:0, tslot:6, thread:0, ctxid:0, blk:0, sublk:0, subErrType:4.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_core_proc.cc][LINE:312]
       Kernel task happen error, retCode=0x31, [vector core exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1539]
       AIV Kernel happen error, retCode=0x31.[FUNC:GetError][FILE:stream.cc][LINE:1190]
       [AIC_INFO] after execute:args print end[FUNC:GetError][FILE:stream.cc][LINE:1190]
       Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=1, flip_num=0, fault kernel_name=compute_src2dst_triton_kernel_0, fault kernel info ext=compute_src2dst_triton_kernel, program id=0, hash=15011427645934493387.[FUNC:GetError][FILE:stream.cc][LINE:1190]
       rtStreamSynchronize execute failed, reason=[suspect remote error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
       synchronize stream failed, runtime result = 507057[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
 (function copy_between_host_and_device_opapi)
Traceback (most recent call last):
  File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 119, in <module>
    run_and_compare(path)  # 对比cuda和triton-ascend的输出
  File "/home/coder/workspace/sglang-test/compute_src2dst_triton_kernel/test_compute_src2dst_triton_kernel.py", line 74, in run_and_compare
    print(">> reorder_ids:", reorder_ids.cpu().numpy())
RuntimeError: ACL stream synchronize failed, error code:507057
[W722 01:57:23.471890626 compiler_depend.ts:526] Warning: NPU warning, error code is 507057[Error]: .
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[suspect remote error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 18805] 2025-07-22-01:57:23.432.688 wait for compute device to finish failed, runtime result = 507057.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function npuSynchronizeUsedDevices)
[W722 01:57:23.473579701 compiler_depend.ts:508] Warning: NPU warning, error code is 507057[Error]: .
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[suspect remote error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 18805] 2025-07-22-01:57:23.434.418 wait for compute device to finish failed, runtime result = 507057.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function npuSynchronizeDevice)
[W722 01:57:23.474899271 compiler_depend.ts:151] Warning: NPU warning, error code is 507057[Error]: .
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[suspect remote error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 18805] 2025-07-22-01:57:23.435.817 wait for compute device to finish failed, runtime result = 507057.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function empty_cache)
[W722 01:57:23.476100720 compiler_depend.ts:508] Warning: NPU warning, error code is 507057[Error]: .
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[suspect remote error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 18805] 2025-07-22-01:57:23.437.083 wait for compute device to finish failed, runtime result = 507057.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function npuSynchronizeDevice)
[W722 01:57:23.477314598 compiler_depend.ts:151] Warning: NPU warning, error code is 507057[Error]: .
EH9999: Inner Error!
        rtDeviceSynchronizeWithTimeout execute failed, reason=[suspect remote error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
EH9999: [PID: 18805] 2025-07-22-01:57:23.438.337 wait for compute device to finish failed, runtime result = 507057.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        TraceBack (most recent call last):
 (function empty_cache)
[ERROR] 2025-07-22-01:57:25 (PID:18805, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
