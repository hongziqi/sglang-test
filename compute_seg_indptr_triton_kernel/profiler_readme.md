## NPU
>>无 autotune
### NORMAL
avg time: 10.762us

>> op_statistic.csv

OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
compute_seg_indptr_triton_kernel,AI_VECTOR_CORE,30,322.866,10.0,10.762,11.3,100.0


## GPU
>>无 autotune
### NORMAL
avg time: 5.000us

[INFO] Profiling compute_seg_indptr_impl with 41 iterations...
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    compute_seg_indptr_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     150.013us       100.00%     150.013us       5.000us            30  
             compute_seg_indptr_impl         0.00%       0.000us         0.00%       0.000us       0.000us     150.013us       100.00%     150.013us       5.000us            30  
                       ProfilerStep*        26.85%     456.286us        99.37%       1.689ms      56.284us       0.000us         0.00%       0.000us       0.000us            30  
             compute_seg_indptr_impl        62.91%       1.069ms        72.52%       1.232ms      41.074us       0.000us         0.00%       0.000us       0.000us            30  
                      cuLaunchKernel         9.61%     163.265us         9.61%     163.265us       5.442us       0.000us         0.00%       0.000us       0.000us            30  
               cudaDeviceSynchronize         0.63%      10.746us         0.63%      10.746us      10.746us       0.000us         0.00%       0.000us       0.000us             1  
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.699ms
Self CUDA time total: 150.013us


## Result
| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| memcpy_triton_kernel | -:5.000us |   -:10.762    | 46.45% | - | - | - |
