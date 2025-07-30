## NPU
### AUTOTUNE
BLOCK_SIZE: 128, avg time: 12.674us

Triton autotuning for function pre_reorder_triton_kernel finished after 28.40s; best config selected: BLOCK_SIZE: 128, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
pre_reorder_triton_kernel,AI_VECTOR_CORE,30,380.207,11.2,12.674,14.86,100.0


### NORMAL
BLOCK_SIZE: 512, avg time: 12.708us

>> op_statistic.csv

OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
pre_reorder_triton_kernel,AI_VECTOR_CORE,30,381.247,10.92,12.708,14.56,100.0


## GPU
### AUTOTUNE
BLOCK_SIZE: 1024, avg time: 3.030us

[INFO] Profiling pre_reorder_impl with 53 iterations...
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    pre_reorder_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      90.908us       100.00%      90.908us       3.030us            30  
             pre_reorder_impl         0.00%       0.000us         0.00%       0.000us       0.000us      90.908us       100.00%      90.908us       3.030us            30  
                ProfilerStep*        19.80%     457.591us        99.70%       2.305ms      76.822us       0.000us         0.00%       0.000us       0.000us            30  
             pre_reorder_impl        72.62%       1.679ms        79.91%       1.847ms      61.569us       0.000us         0.00%       0.000us       0.000us            30  
               cuLaunchKernel         7.29%     168.442us         7.29%     168.442us       5.615us       0.000us         0.00%       0.000us       0.000us            30  
        cudaDeviceSynchronize         0.30%       6.827us         0.30%       6.827us       6.827us       0.000us         0.00%       0.000us       0.000us             1  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.311ms
Self CUDA time total: 90.908us

[INFO] Profiling results saved to cuda_profiling_results/pre_reorder_impl_trace.json


### NORMAL
BLOCK_SIZE: 512, avg time: 3.710us

[INFO] Profiling pre_reorder_impl with 53 iterations...
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    pre_reorder_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     111.294us       100.00%     111.294us       3.710us            30  
             pre_reorder_impl         0.00%       0.000us         0.00%       0.000us       0.000us     111.294us       100.00%     111.294us       3.710us            30  
                ProfilerStep*        23.08%     446.471us        99.35%       1.922ms      64.066us       0.000us         0.00%       0.000us       0.000us            30  
             pre_reorder_impl        68.12%       1.318ms        76.27%       1.476ms      49.184us       0.000us         0.00%       0.000us       0.000us            30  
               cuLaunchKernel         8.16%     157.790us         8.16%     157.790us       5.260us       0.000us         0.00%       0.000us       0.000us            30  
        cudaDeviceSynchronize         0.65%      12.558us         0.65%      12.558us      12.558us       0.000us         0.00%       0.000us       0.000us             1  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.935ms
Self CUDA time total: 111.294us

[INFO] Profiling results saved to cuda_profiling_results/pre_reorder_impl_trace.json


## Result
| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| pre_reorder_triton_kernel | 512:3.710us |   512:12.708us    | 29.19% | 1024:3.030us | 128:12.674us | 23.9% |
