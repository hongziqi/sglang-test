## NPU
### AUTOTUNE



## GPU
### AUTOTUNE
page_size: 512(会变), avg time: 1.521us

==================== Test AutoTune First ====================
Triton autotuning for function alloc_decode_kernel finished after 0.75s; best config selected: page_size: 512, num_warps: 4, num_ctas: 1, num_stages: 2, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
==================== Test AutoTune Done ====================

==================== Profiling the Triton kernel start, Autotune:True ====================
[INFO] Profiling alloc_decode_triton_launcher with 53 iterations...
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
             alloc_decode_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      45.632us       100.00%      45.632us       1.521us            30  
    alloc_decode_triton_launcher         0.00%       0.000us         0.00%       0.000us       0.000us      45.632us       100.00%      45.632us       1.521us            30  
                   ProfilerStep*        18.20%     552.961us        99.62%       3.026ms     100.882us       0.000us         0.00%       0.000us       0.000us            30  
    alloc_decode_triton_launcher        72.75%       2.210ms        81.42%       2.473ms      82.450us       0.000us         0.00%       0.000us       0.000us            30  
                  cuLaunchKernel         8.67%     263.390us         8.67%     263.390us       8.780us       0.000us         0.00%       0.000us       0.000us            30  
           cudaDeviceSynchronize         0.38%      11.658us         0.38%      11.658us      11.658us       0.000us         0.00%       0.000us       0.000us             1  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.038ms
Self CUDA time total: 45.632us

[INFO] Profiling results saved to cuda_profiling_results/alloc_decode_triton_launcher_trace.json
==================== Profiling the Triton kernel done, Autotune:True ====================

## Result
(原始输入未提供BLOCK_SIZE,当前仅测试autotune下的耗时)

| 算子名称                 | GPU描述:耗时 | NPU描述:耗时 | 比例 | autotune描述:GPU耗时 | autotune描述:NPU耗时  | autotune比例 |
| -------------------- | ----- | ----- | ----- | ----- | --- | ---------- | 
| alloc_decode_kernel | -:- |   -:-    | - | 512:1.521us |  |  |
