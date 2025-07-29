## NPU
### AUTOTUNE
BLOCK_SIZE: 4096, avg time: 1.882us

Triton autotuning for function memcpy_triton_kernel finished after 16.82s; best config selected: BLOCK_SIZE: 4096, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv

OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
memcpy_triton_kernel,AI_VECTOR_CORE,30,56.461,1.74,1.882,2.04,100.0


### NORMAL
BLOCK_SIZE: 8192, avg time: 3.286us

>> op_statistic.csv

OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
memcpy_triton_kernel,AI_VECTOR_CORE,30,98.582,1.4,3.286,4.78,100.0


## GPU
### AUTOTUNE
BLOCK_SIZE: 8192, avg time: 2.218us

>>> Profiling Started

[INFO] Profiling memcpy_triton_kernel_impl with 41 iterations...
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
         memcpy_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      66.526us       100.00%      66.526us       2.218us            30  
    memcpy_triton_kernel_impl         0.00%       0.000us         0.00%       0.000us       0.000us      66.526us       100.00%      66.526us       2.218us            30  
                ProfilerStep*        20.66%     466.564us        99.70%       2.252ms      75.053us       0.000us         0.00%       0.000us       0.000us            30  
    memcpy_triton_kernel_impl        71.44%       1.613ms        79.04%       1.785ms      59.501us       0.000us         0.00%       0.000us       0.000us            30  
               cuLaunchKernel         7.61%     171.752us         7.61%     171.752us       5.725us       0.000us         0.00%       0.000us       0.000us            30  
        cudaDeviceSynchronize         0.30%       6.674us         0.30%       6.674us       6.674us       0.000us         0.00%       0.000us       0.000us             1  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.258ms
Self CUDA time total: 66.526us


### NORMAL
BLOCK_SIZE: 8192, avg time: 2.209us

[INFO] Profiling memcpy_triton_kernel_impl with 41 iterations...
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
         memcpy_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      66.270us       100.00%      66.270us       2.209us            30  
    memcpy_triton_kernel_impl         0.00%       0.000us         0.00%       0.000us       0.000us      66.270us       100.00%      66.270us       2.209us            30  
                ProfilerStep*        20.05%     448.383us        99.88%       2.233ms      74.447us       0.000us         0.00%       0.000us       0.000us            30  
    memcpy_triton_kernel_impl        68.37%       1.529ms        79.83%       1.785ms      59.501us       0.000us         0.00%       0.000us       0.000us            30  
               cuLaunchKernel         7.43%     166.236us         7.43%     166.236us       5.541us       0.000us         0.00%       0.000us       0.000us            30  
        cudaDeviceSynchronize         4.14%      92.596us         4.14%      92.596us       2.987us       0.000us         0.00%       0.000us       0.000us            31  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.236ms
Self CUDA time total: 66.270us


## Result
| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| memcpy_triton_kernel | 8192:2.209 |   8192:3.286    | 67.22% | 8192:2.218 | 4096:1.882 | 117.85% |