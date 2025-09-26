import triton
import triton.language as tl
import torch


@triton.jit
def triton_load_store_sge_mask(in_ptr0, out_ptr0, threshold: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index >= threshold
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + x_index, tmp2, xmask)


if __name__ == "__main__":
    dtype, shape, ncore, xblock, xblock_sub = torch.float32, (2, 4096, 8), 2, 32768, 1024
    x0 = torch.randn(shape, dtype=dtype).npu()
    threshold = x0.numel() // 2
    y_ref = x0[threshold:]
    y_cal = torch.zeros_like(x0).npu()
    triton_load_store_sge_mask[(ncore, )](x0, y_cal, threshold, xblock, xblock_sub)
    torch.testing.assert_close(y_cal[threshold:], y_ref)
    print("test pass!")

    # Traceback (most recent call last):
    # File "/home/coder/workspace/triton-ascend/../sglang-test/other_test/test_load_store_sge_mask.py", line 23, in <module>
    #     triton_load_store_sge_mask[(ncore, )](x0, y_cal, threshold, xblock, xblock_sub)
    # File "/home/coder/workspace/triton-ascend/triton/runtime/jit.py", line 331, in <lambda>
    #     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
    # File "/home/coder/workspace/triton-ascend/triton/runtime/jit.py", line 635, in run
    #     kernel = self.compile(
    # File "/home/coder/workspace/triton-ascend/triton/compiler/compiler.py", line 297, in compile
    #     raise MLIRCompilationError(stage_name, error_detail)
    # triton.compiler.errors.MLIRCompilationError: 
    # ///------------------[ERROR][Triton][BEG]------------------
    # [ConvertLinalgRToBinary] encounters error:
    # LLVM ERROR: constant int value is not created
    # PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
    # Stack dump:
    # 0.      Program arguments: /home/coder/shared/bisheng_toolkit_0917/bishengir/bin/bishengir-compile /tmp/tmpxncm9w00/kernel.ttadapter.mlir --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true -o /tmp/tmpxncm9w00/kernel
    # ///------------------[ERROR][Triton][END]------------------

    # [ERROR] 2025-09-18-07:10:38 (PID:1327020, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
