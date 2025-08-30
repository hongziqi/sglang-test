import torch
import os
import inspect
from typing import List


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
    skip_first = 20
    wait = 0
    warmup = 3
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
            prof.step()
        torch.cuda.synchronize()
    
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
    skip_first = 20
    wait = 0
    warmup = 3
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


def print_data_info(data):
    """
    打印数据的形状、类型和前10个元素
    :param data: 数据字典 或者 tuple
    """
    if isinstance(data, tuple):
        data_items = enumerate(data)
    elif isinstance(data, dict):
        data_items = data.items()
    else:
        raise TypeError("Data must be a dictionary or a tuple.")
    for key, value in data_items:
            if isinstance(value, torch.Tensor):
                print(f">> {key}:")
                print(f" Shape: {value.shape}")
                print(f" Dtype: {value.dtype}")
                print(f" Device: {value.device}")
                # 打印前10个元素
                print(f" First 10 elements: {value.flatten()[:10].tolist()}")
            elif isinstance(value, int):
                print(f">> {key}: {value}")
            else:
                print(f">> {key}: {value}")


def print_real_data(src_path: str):
    """
    打印真实数据的形状、类型和前10个元素
    :param src_path: 数据文件路径
    """
    try:
        data = torch.load(src_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"File {src_path} not found. Please run the test to generate it.")
        return

    print("\n[REAL DATA INFO]")
    print_data_info(data)



def run_and_compare_real_data_npu(
        triton_kernel_impl,  # Triton 内核实现函数
        src_path: str,
        expected_path: str,
        key_mapping: dict,  # 数据文件参数名与内核参数名的映射
        accuracy: bool = True,  # 是否检查精度
        accuracy_dict: List[str] = None,  # 精度检查的键列表
        autotune: bool = False,  # 是否自动调优
        profiling: bool = False,  # 是否进行性能分析
        USE_BLOCK_SIZE: bool = False,  # 是否使用 block_size
        block_size: int = 8192  # 默认的 BLOCK_SIZE
    ):
    """
    通用的 Triton kernel 测试函数，适配多个内核。
    :param triton_kernel_impl: Triton 内核实现函数
    :param src_path: 输入数据路径
    :param expected_path: 期望输出数据路径
    :param key_mapping: 数据文件参数名与内核参数名的映射 {内核参数： 数据文件参数名}
    :param accuracy: 是否检查精度
    :param accuracy_dict: 精度检查的键列表(内核参数)
    :param autotune: 是否启用自动调优
    :param profiling: 是否进行性能分析
    :param USE_BLOCK_SIZE: 是否使用自定义 BLOCK_SIZE
    :param block_size: 自定义 BLOCK_SIZE 的值
    """
    print(f"[DEBUG] KERLNEL NAME: {triton_kernel_impl.__name__}")
    try:
        data = torch.load(src_path, map_location=torch.device('cpu'))
        expected_data = torch.load(expected_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"File {src_path} or {expected_path} not found. Please run the test to generate it.")
        return
    # print(f"\n[REAL DATA INFO]")
    # print_data_info(data)

    # 将输入数据加载到 NPU
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.npu()

    # 动态解析内核参数
    key_mapping = key_mapping or {}  # 如果没有提供映射表，则使用空字典
    # 获取内核函数的参数顺序
    param_order = list(inspect.signature(triton_kernel_impl).parameters.keys())
    kernel_args = {
        param: data[key_mapping.get(param, param)] if key_mapping.get(param, param) in data else None
        for param in param_order
    }
    kernel_args["autotune"] = autotune  # 添加 autotune 参数
    # 打印内核参数
    print("\n[Load Kernel Arguments]")
    print_data_info(kernel_args)

    # 检查精度
    if accuracy:
        print(f"\n{'='*20} Checking accuracy start... {'='*20}")
        if not accuracy_dict:
            print(">>> No accuracy check keys provided, skipping accuracy check...")
            print("Please provide accuracy_dict to enable accuracy checks.")
        print("\n>>> Running kernel for accuracy check...")
        triton_kernel_impl(**kernel_args)
        torch.npu.synchronize()

        for key in accuracy_dict:
            if key in kernel_args:
                output_tensor = kernel_args[key]
                expected_tensor = expected_data[key_mapping.get(key, key)].npu()
                print(f">>> Checking accuracy for ({key}):")
                check_accuracy(output_tensor, expected_tensor)
        print(f"{'='*20} Checking accuracy done. {'='*20}")

    # 自动调优测试
    if autotune:
        print(f"\n{'='*20} Test AutoTune First {'='*20}")
        triton_kernel_impl(**kernel_args)
        print(f"{'='*20} Test AutoTune Done {'='*20}")

    # 使用自定义 BLOCK_SIZE
    if USE_BLOCK_SIZE and not autotune and kernel_args.get("BLOCK_SIZE") is not None:
        kernel_args["BLOCK_SIZE"] = block_size
        print(f"\n>>> Using custom BLOCK_SIZE: {block_size}")
    
    # 性能分析
    if profiling:
        # 按顺序提取参数值
        args = tuple(kernel_args[param] for param in param_order if param in kernel_args)
        # print_data_info(args)
        print(f"\n{'='*20} Profiling the Triton kernel start, Autotune:{autotune} {'='*20}")
        profiling_test_npu(
            triton_kernel_impl,
            args=args,
        )
        print(f"{'='*20} Profiling the Triton kernel done, Autotune:{autotune} {'='*20}")


def run_and_compare_real_data_cuda(
        triton_kernel_impl,  # Triton 内核实现函数
        src_path: str,
        expected_path: str,
        key_mapping: dict,  # 数据文件参数名与内核参数名的映射
        save_output: bool = False,  # 是否保存输出数据
        autotune: bool = False,  # 是否自动调优
        profiling: bool = False,  # 是否进行性能分析
        USE_BLOCK_SIZE: bool = False,  # 是否使用 block_size
        block_size: int = 8192  # 默认的 BLOCK_SIZE
    ):
    """
    通用的 Triton kernel 测试函数，适配多个内核。
    :param triton_kernel_impl: Triton 内核实现函数
    :param src_path: 输入数据路径
    :param expected_path: 期望输出数据路径
    :param key_mapping: 数据文件参数名与内核参数名的映射 {内核参数： 数据文件参数名}
    :param save_output: 是否保存输出数据
    :param autotune: 是否启用自动调优
    :param profiling: 是否进行性能分析
    :param USE_BLOCK_SIZE: 是否使用自定义 BLOCK_SIZE
    :param block_size: 自定义 BLOCK_SIZE 的值
    """
    try:
        data = torch.load(src_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"File {src_path} or {expected_path} not found. Please run the test to generate it.")
        return

    print(f"\n[REAL DATA INFO({triton_kernel_impl.__name__})]")
    print_data_info(data)

    # 将输入数据加载到 CUDA
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.cuda()

    # 动态解析内核参数
    key_mapping = key_mapping or {}  # 如果没有提供映射表，则使用空字典
    # 获取内核函数的参数顺序
    param_order = list(inspect.signature(triton_kernel_impl).parameters.keys())
    kernel_args = {
        param: data[key_mapping.get(param, param)] if key_mapping.get(param, param) in data else None
        for param in param_order
    }
    kernel_args["autotune"] = autotune  # 添加 autotune 参数
    # 打印内核参数
    print("\n[Load Kernel Arguments]")
    print_data_info(kernel_args)

    if save_output:
        triton_kernel_impl(**kernel_args)
        # 根据输入文件的文件参数录入文件 key_mapping: {内核参数：文件参数}
        save_args = {key_mapping.get(param, param): value for param, value in kernel_args.items() if key_mapping.get(param, param) in data}
        print("\n[Save Output Data]")
        print_data_info(save_args)
        torch.save(save_args, expected_path)

    # 自动调优测试
    if autotune:
        print(f"\n{'='*20} Test AutoTune First {'='*20}")
        triton_kernel_impl(**kernel_args)
        print(f"{'='*20} Test AutoTune Done {'='*20}")
    
    # 使用自定义 BLOCK_SIZE
    if USE_BLOCK_SIZE and not autotune and kernel_args.get("BLOCK_SIZE") is not None:
        kernel_args["BLOCK_SIZE"] = block_size
        print(f"\n>>> Using custom BLOCK_SIZE: {block_size}")
    
    # 性能分析
    if profiling:
        # 按顺序提取参数值
        args = tuple(kernel_args[param] for param in param_order if param in kernel_args)
        print(f"\n{'='*20} Profiling the Triton kernel start, Autotune:{autotune} {'='*20}")
        profiling_test_cuda(
            triton_kernel_impl,
            args=args,
        )
        print(f"{'='*20} Profiling the Triton kernel done, Autotune:{autotune} {'='*20}")
