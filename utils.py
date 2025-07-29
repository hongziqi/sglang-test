import torch
import os


def check_accuracy(output: torch.Tensor, expected: torch.Tensor):
    assert output.shape == expected.shape, f"Shape mismatch: {output.shape} vs {expected.shape}"
    
    # 根据 dtype 自动判定阈值
    dtype = expected.dtype
    if dtype == torch.float16:
        print(">>> Compare Type: float16")
        rtol, atol, max_fail_ratio = 1e-3, 1e-3, 1e-3  # 双千分之一
    elif dtype == torch.bfloat16:
        print(">>> Compare Type: bfloat16")
        rtol, atol, max_fail_ratio = 5e-3, 5e-3, 5e-3  # 双千分之五
    elif dtype == torch.float32:
        print(">>> Compare Type: float32")
        rtol, atol, max_fail_ratio = 1e-4, 1e-4, 1e-4  # 双万分之一
    elif dtype in [torch.int8, torch.uint8, torch.int32, torch.uint32, torch.int64, torch.uint64]:
        print(">>> Compare Type: int")
        rtol, atol, max_fail_ratio = 0, 0, 0  # 整数类型不允许误差
    else:
        raise ValueError(f"Unsupported dtype for accuracy check: {dtype}")

    # 计算误差
    abs_diff = (output - expected).abs()
    rel_diff = abs_diff / (expected.abs() + 1e-6)
    fail_mask = (abs_diff > atol) & (rel_diff > rtol)

    total = output.numel()
    fail = fail_mask.sum().item()
    fail_ratio = fail / total

    # 打印最大误差点
    max_abs = abs_diff.max().item()
    if max_abs > 0:
        max_idx_flat = torch.argmax(abs_diff)  # 不使用 .item()
        max_idx = torch.unravel_index(max_idx_flat, output.shape)  # 适配多维张量
        print(f"Max diff at {max_idx}: test={output[max_idx].item()}, "
              f"ref={expected[max_idx].item()}, "
              f"abs={abs_diff[max_idx].item()}, rel={rel_diff[max_idx].item()}")

    # 判断是否精度达标
    if fail_ratio <= max_fail_ratio:
        print(f"精度达标 ({fail}/{total}, {fail_ratio:.6%} <= {max_fail_ratio:.6%})")
    else:
        print(f"精度不达标 ({fail}/{total}, {fail_ratio:.6%} > {max_fail_ratio:.6%})")
        idx_list = torch.nonzero(fail_mask)[:10]  # 获取前10个失败点
        for idx in idx_list.tolist():
            idx_tuple = tuple(idx)  # 转换为多维索引
            print(f"{idx_tuple}: test={output[idx_tuple].item():.6f}, "
                  f"ref={expected[idx_tuple].item():.6f}, "
                  f"diff={abs_diff[idx_tuple].item():.6f}, rel={rel_diff[idx_tuple].item():.6f}")

    return fail_ratio


def profiling_test_cuda(fn_triton, args=(), result_dir="cuda_profiling_results"):
    """
    用于测试triton kernel的profiling功能
    :param fn_triton: triton kernel函数
    :param args: 函数参数
    :param activities: 需要记录的活动列表
    :param result_path: 结果保存路径
    """
    skip_first = 10
    wait = 0
    warmup = 1
    active = 30
    repeat = 1
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    from torch.profiler import profile, record_function, ProfilerActivity
    LOOP = skip_first + (wait + warmup + active) * repeat
    print(f"[INFO] Profiling {fn_triton.__name__} with {LOOP} iterations...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(LOOP):
            with record_function(fn_triton.__name__):
                fn_triton(*args)
                torch.cuda.synchronize()
            prof.step()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    file_name_without_ext = os.path.splitext(fn_triton.__name__)[0]
    new_path = f"{file_name_without_ext}_trace.json"
    save_path = os.path.join(result_dir, new_path)

    prof.export_chrome_trace(save_path)

    print(f"[INFO] Profiling results saved to {save_path}")


def profiling_test_npu(fn_triton, args=(), result_dir="npu_profiling_results"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    import torch_npu
    skip_first = 10
    wait = 0
    warmup = 1
    active = 30
    repeat = 1
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,
                                                 skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config) as prof:
        stream.synchronize()
        for i in range(skip_first + (wait + warmup + active) * repeat):
            fn_triton(*args)
            prof.step()
        stream.synchronize()
