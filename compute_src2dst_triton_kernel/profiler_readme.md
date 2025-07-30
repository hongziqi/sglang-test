## NPU
### AUTOTUNE
BLOCK_SIZE: 128, avg time: 40.017us

Triton autotuning for function compute_src2dst_triton_kernel finished after 28.39s; best config selected: BLOCK_SIZE: 128, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
compute_src2dst_triton_kernel,AI_VECTOR_CORE,30,1200.504,38.701,40.017,41.981,100.0


### NORMAL
BLOCK_SIZE: 512, avg time: 114.178us

>> op_statistic.csv

OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
compute_src2dst_triton_kernel,AI_VECTOR_CORE,30,3425.349,109.042,114.178,119.482,100.0



## GPU
### AUTOTUNE
BLOCK_SIZE: 会变
BLOCK_SIZE: 1024, avg time: 2.366us

==================== Profiling the Triton kernel start, Autotune:True ====================
[INFO] Profiling compute_src2dst_impl with 53 iterations...
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    compute_src2dst_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      70.976us       100.00%      70.976us       2.366us            30  
             compute_src2dst_impl         0.00%       0.000us         0.00%       0.000us       0.000us      70.976us       100.00%      70.976us       2.366us            30  
                    ProfilerStep*        22.24%     442.876us        99.69%       1.985ms      66.171us       0.000us         0.00%       0.000us       0.000us            30  
             compute_src2dst_impl        69.71%       1.388ms        77.45%       1.542ms      51.408us       0.000us         0.00%       0.000us       0.000us            30  
                   cuLaunchKernel         7.74%     154.122us         7.74%     154.122us       5.137us       0.000us         0.00%       0.000us       0.000us            30  
            cudaDeviceSynchronize         0.31%       6.264us         0.31%       6.264us       6.264us       0.000us         0.00%       0.000us       0.000us             1  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.991ms
Self CUDA time total: 70.976us

[INFO] Profiling results saved to cuda_profiling_results/compute_src2dst_impl_trace.json


### NORMAL
BLOCK_SIZE: 512, avg time: 2.824us

[INFO] Profiling compute_src2dst_impl with 53 iterations...
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    compute_src2dst_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      84.733us       100.00%      84.733us       2.824us            30  
             compute_src2dst_impl         0.00%       0.000us         0.00%       0.000us       0.000us      84.733us       100.00%      84.733us       2.824us            30  
                    ProfilerStep*        26.05%     429.453us        99.49%       1.640ms      54.680us       0.000us         0.00%       0.000us       0.000us            30  
             compute_src2dst_impl        64.07%       1.056ms        73.44%       1.211ms      40.365us       0.000us         0.00%       0.000us       0.000us            30  
                   cuLaunchKernel         9.38%     154.623us         9.38%     154.623us       5.154us       0.000us         0.00%       0.000us       0.000us            30  
            cudaDeviceSynchronize         0.51%       8.383us         0.51%       8.383us       8.383us       0.000us         0.00%       0.000us       0.000us             1  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.649ms
Self CUDA time total: 84.733us

[INFO] Profiling results saved to cuda_profiling_results/compute_src2dst_impl_trace.json


## Result
| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| compute_src2dst_triton_kernel | 512:2.824us |   512:114.178us    | 2.47% | 1024:2.366us | 128:40.017us | 5.91% |
