## NPU
### AUTOTUNE
BLOCK_SIZE: 8192, avg time: 2.819us

Triton autotuning for function deepep_permute_triton_kernel finished after 118.05s; best config selected: BLOCK_SIZE: 8192, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;

>> op_statistic.csv
OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
deepep_permute_triton_kernel,AI_VECTOR_CORE,30,84.582,2.56,2.819,3.041,100


## GPU
### AUTOTUNE
BLOCK_SIZE: 8192, avg time: 3.494us

==================== Test AutoTune First ====================
Triton autotuning for function deepep_permute_triton_kernel finished after 0.52s; best config selected: BLOCK_SIZE: 8192, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
==================== Test AutoTune Done ====================

==================== Profiling the Triton kernel start, Autotune:True ====================
[INFO] Profiling deepep_permute_impl with 53 iterations...
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    deepep_permute_triton_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     104.830us       100.00%     104.830us       3.494us            30  
             deepep_permute_impl         0.00%       0.000us         0.00%       0.000us       0.000us     104.830us       100.00%     104.830us       3.494us            30  
                   ProfilerStep*        18.65%     457.940us        99.71%       2.449ms      81.618us       0.000us         0.00%       0.000us       0.000us            30  
             deepep_permute_impl        74.06%       1.819ms        81.06%       1.991ms      66.353us       0.000us         0.00%       0.000us       0.000us            30  
                  cuLaunchKernel         7.00%     171.950us         7.00%     171.950us       5.732us       0.000us         0.00%       0.000us       0.000us            30  
           cudaDeviceSynchronize         0.29%       7.159us         0.29%       7.159us       7.159us       0.000us         0.00%       0.000us       0.000us             1  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.456ms
Self CUDA time total: 104.830us

[INFO] Profiling results saved to cuda_profiling_results/deepep_permute_impl_trace.json
==================== Profiling the Triton kernel done, Autotune:True ====================


## Result
(原始输入未提供BLOCK_SIZE,当前仅测试autotune下的耗时)

| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| deepep_permute_triton_kernel | -:- |   -:-    | - | 8192:3.494us | 8192:2.819us | 123.94% |
