## NPU
### AUTOTUNE
BLOCK_SIZE: 4096, avg time: 1.885us

Triton autotuning for function memcpy_triton_kernel finished after 16.82s; best config selected: BLOCK_SIZE: 4096, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
memcpy_triton_kernel,AI_VECTOR_CORE,30,56.541,1.76,1.885,2.02,100.0

### NORMAL
BLOCK_SIZE: 8192, avg time: 3.286us
>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
memcpy_triton_kernel,AI_VECTOR_CORE,30,98.582,1.4,3.286,4.78,100.0


## GPU
### AUTOTUNE
BLOCK_SIZE: 8192, avg time: 1.812us
[INFO] Profiling memcpy_triton_kernel_impl with 41 iterations...
Autotuning kernel memcpy_triton_kernel with config BLOCK_SIZE: 4096, num_warps: 4, num_ctas: 1, num_stages: 3, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None
Autotuning kernel memcpy_triton_kernel with config BLOCK_SIZE: 8192, num_warps: 4, num_ctas: 1, num_stages: 3, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None
Triton autotuning for function memcpy_triton_kernel finished after 0.52s; best config selected: BLOCK_SIZE: 8192, num_warps: 4, num_ctas: 1, num_stages: 3, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
         memcpy_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      54.368us       100.00%      54.368us       1.812us            30  
    memcpy_triton_kernel_impl         0.00%       0.000us         0.00%       0.000us       0.000us      54.368us       100.00%      54.368us       1.812us            30  
                ProfilerStep*        16.99%     463.162us        99.90%       2.724ms      90.798us       0.000us         0.00%       0.000us       0.000us            30  
    memcpy_triton_kernel_impl        71.34%       1.945ms        82.91%       2.261ms      75.359us       0.000us         0.00%       0.000us       0.000us            30  
               cuLaunchKernel         7.53%     205.274us         7.53%     205.274us       6.842us       0.000us         0.00%       0.000us       0.000us            30  
        cudaDeviceSynchronize         4.14%     112.976us         4.14%     112.976us       3.644us       0.000us         0.00%       0.000us       0.000us            31  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.727ms
Self CUDA time total: 54.368us

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
| memcpy_triton_kernel | 8192:2.209 |   8192:3.286    |   | 8192:1.812 | 4096:1.885 |  |