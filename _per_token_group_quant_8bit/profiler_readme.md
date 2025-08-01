## NPU
### AUTOTUNE
BLOCK_SIZE: 128, num_warps: 4, num_stages: 2, avg time: 296.438us

Triton autotuning for function _per_token_group_quant_8bit finished after 23.93s; best config selected: BLOCK: 128, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
_per_token_group_quant_8bit,AI_VECTOR_CORE,30,8893.138,256.745,296.438,350.647,100.0



### NORMAL
BLOCK_SIZE: 128, num_warps: 1, num_stages: 1, avg time: 301.299us

>> op_statistic.csv

OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
_per_token_group_quant_8bit,AI_VECTOR_CORE,30,9038.961,256.765,301.299,350.967,100.0




## GPU
### AUTOTUNE
BLOCK_SIZE: 会变
BLOCK_SIZE: 128, num_warps: 4, num_stages: 3, avg time: 5.325us

Triton autotuning for function _per_token_group_quant_8bit finished after 0.83s; best config selected: BLOCK: 128, num_warps: 4, num_ctas: 1, num_stages: 3, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

[INFO] Profiling triton_per_token_group_quant_8bit_impl with 53 iterations...
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
               _per_token_group_quant_8bit         0.00%       0.000us         0.00%       0.000us       0.000us     159.740us       100.00%     159.740us       5.325us            30  
    triton_per_token_group_quant_8bit_impl         0.00%       0.000us         0.00%       0.000us       0.000us     159.740us       100.00%     159.740us       5.325us            30  
                             ProfilerStep*        21.02%     461.037us        99.68%       2.186ms      72.880us       0.000us         0.00%       0.000us       0.000us            30  
    triton_per_token_group_quant_8bit_impl        71.27%       1.563ms        78.66%       1.725ms      57.512us       0.000us         0.00%       0.000us       0.000us            30  
                            cuLaunchKernel         7.39%     162.010us         7.39%     162.010us       5.400us       0.000us         0.00%       0.000us       0.000us            30  
                     cudaDeviceSynchronize         0.32%       7.100us         0.32%       7.100us       7.100us       0.000us         0.00%       0.000us       0.000us             1  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.193ms
Self CUDA time total: 159.740us

[INFO] Profiling results saved to cuda_profiling_results/triton_per_token_group_quant_8bit_impl_trace.json


### NORMAL
BLOCK_SIZE: 128, num_warps: 1, num_stages: 1, avg time: 6.374us

[INFO] Profiling triton_per_token_group_quant_8bit_impl with 53 iterations...
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
               _per_token_group_quant_8bit         0.00%       0.000us         0.00%       0.000us       0.000us     191.227us       100.00%     191.227us       6.374us            30  
    triton_per_token_group_quant_8bit_impl         0.00%       0.000us         0.00%       0.000us       0.000us     191.227us       100.00%     191.227us       6.374us            30  
                             ProfilerStep*        24.04%     448.966us        99.44%       1.857ms      61.894us       0.000us         0.00%       0.000us       0.000us            30  
    triton_per_token_group_quant_8bit_impl        66.87%       1.249ms        75.40%       1.408ms      46.929us       0.000us         0.00%       0.000us       0.000us            30  
                            cuLaunchKernel         8.53%     159.296us         8.53%     159.296us       5.310us       0.000us         0.00%       0.000us       0.000us            30  
                     cudaDeviceSynchronize         0.56%      10.388us         0.56%      10.388us      10.388us       0.000us         0.00%       0.000us       0.000us             1  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.867ms
Self CUDA time total: 191.227us

[INFO] Profiling results saved to cuda_profiling_results/triton_per_token_group_quant_8bit_impl_trace.json

## Result
| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| _per_token_group_quant_8bit | 128,1,1:6.374us | 128,1,1:301.299us | 2.11% | 128,4,3:5.325us | 128,4,2:296.438us | 1.79% |
