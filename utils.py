import torch
import torch_npu


def check_accuracy(output: torch.Tensor, expected: torch.Tensor):
    # 根据 dtype 自动判定阈值
    dtype = expected.dtype
    if dtype == torch.float16:
        print(">>> Compare Type: float16")
        rtol, atol, max_fail_ratio = 1e-3, 1e-3, 1e-3  # 双千分之一
    elif dtype == torch.float32:
        print(">>> Compare Type: float32")
        rtol, atol, max_fail_ratio = 1e-4, 1e-4, 1e-4  # 双万分之一
    elif dtype == torch.int32:
        print(">>> Compare Type: int32")
        rtol, atol, max_fail_ratio = 1e-3, 1, 1e-3  # 容差为1
    elif dtype in [torch.int8, torch.uint8]:
        print(">>> Compare Type: int8 | uint8")
        rtol, atol, max_fail_ratio = 1e-3, 1, 1e-3     # 容差为1
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
        max_idx_flat = torch.argmax(abs_diff).item()
        i, j = divmod(max_idx_flat, output.shape[1])
        print(f"Max diff at [{i}, {j}]: test={output[i, j].item()}, "
            f"ref={expected[i, j].item()}, "
            f"abs={abs_diff[i, j].item()}, rel={rel_diff[i, j].item()}")

    # 判断是否精度达标
    if fail_ratio <= max_fail_ratio:
        print(f"精度达标 ({fail}/{total}, {fail_ratio:.6%} <= {max_fail_ratio:.6%})")
    else:
        print(f"精度不达标 ({fail}/{total}, {fail_ratio:.6%} > {max_fail_ratio:.6%})")
        idx_list = torch.nonzero(fail_mask)[:10]
        for i, j in idx_list.tolist():
            print(f"[{i},{j}]: test={output[i, j].item():.6f}, "
                  f"ref={expected[i, j].item():.6f}, "
                  f"diff={abs_diff[i, j].item():.6f}, rel={rel_diff[i, j].item():.6f}")

    return fail_ratio