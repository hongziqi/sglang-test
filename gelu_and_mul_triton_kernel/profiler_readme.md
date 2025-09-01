## NPU
### AUTOTUNE
BLOCK_SIZE: 2048, avg time: 4.095us

Triton autotuning for function gelu_and_mul_triton_kernel finished after 29.82s; best config selected: BLOCK_SIZE: 2048, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
gelu_and_mul_triton_kernel,AI_VECTOR_CORE,30,122.861,3.88,4.095,4.3,100


## GPU
### AUTOTUNE
BLOCK_SIZE: 2048, avg time: 2.663us

==================== Test AutoTune First ====================
Triton autotuning for function gelu_and_mul_triton_kernel finished after 0.61s; best config selected: BLOCK_SIZE: 2048, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
==================== Test AutoTune Done ====================

==================== Profiling the Triton kernel start, Autotune:True ====================
[INFO] Profiling gelu_and_mul_triton_launcher with 53 iterations...
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
      gelu_and_mul_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      79.902us       100.00%      79.902us       2.663us            30  
    gelu_and_mul_triton_launcher         0.00%       0.000us         0.00%       0.000us       0.000us      79.902us       100.00%      79.902us       2.663us            30  
                   ProfilerStep*        18.84%     457.824us        99.71%       2.423ms      80.765us       0.000us         0.00%       0.000us       0.000us            30  
    gelu_and_mul_triton_launcher        73.99%       1.798ms        80.87%       1.965ms      65.505us       0.000us         0.00%       0.000us       0.000us            30  
                  cuLaunchKernel         6.88%     167.122us         6.88%     167.122us       5.571us       0.000us         0.00%       0.000us       0.000us            30  
           cudaDeviceSynchronize         0.29%       7.163us         0.29%       7.163us       7.163us       0.000us         0.00%       0.000us       0.000us             1  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.430ms
Self CUDA time total: 79.902us

[INFO] Profiling results saved to cuda_profiling_results/gelu_and_mul_triton_launcher_trace.json
==================== Profiling the Triton kernel done, Autotune:True ====================


## Result
(原始输入未提供BLOCK_SIZE,当前仅测试autotune下的耗时)

| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| gelu_and_mul_triton_kernel | -:- |   -:-    | - | 2048:2.663us | 2048:4.095us | 65.03% |
