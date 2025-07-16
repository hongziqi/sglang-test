Traceback (most recent call last):
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/compiler/compiler.py", line 288, in compile
    next_module = compile_ir(module, metadata)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/backends/ascend/compiler.py", line 423, in <lambda>
    stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/backends/ascend/compiler.py", line 83, in ttir_to_linalg
    ret = subprocess.run(cmd_list, capture_output=True, check=True)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/backends/ascend/triton-adapter-opt', '/tmp/tmpcx8q74h5/kernel.ttir.mlir', '--triton-to-linalg=global-kernel=false named-ops=True', '-o', '/tmp/tmpcx8q74h5/kernel.ttadapter.mlir']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/coder/workspace/sglang-test/deepep_compute_src2dst_triton_kernel/test_deepep_compute_src2dst_triton_kernel.py", line 109, in <module>
    save_inputs_outputs(path)
  File "/home/coder/workspace/sglang-test/deepep_compute_src2dst_triton_kernel/test_deepep_compute_src2dst_triton_kernel.py", line 68, in save_inputs_outputs
    src2dst = deepep_compute_src2dst_impl(
  File "/home/coder/workspace/sglang-test/deepep_compute_src2dst_triton_kernel/test_deepep_compute_src2dst_triton_kernel.py", line 47, in deepep_compute_src2dst_impl
    deepep_compute_src2dst_triton_kernel[grid](
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/runtime/jit.py", line 331, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/runtime/jit.py", line 625, in run
    kernel = self.compile(
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/compiler/compiler.py", line 297, in compile
    raise MLIRCompilationError(stage_name, error_detail)
triton.compiler.errors.MLIRCompilationError: 
///------------------[ERROR][Triton][BEG]------------------
[ConvertTritonIRToLinalgIR] encounters error:
/home/coder/workspace/sglang-test/deepep_compute_src2dst_triton_kernel/test_deepep_compute_src2dst_triton_kernel.py:20:35: error: operand #0 does not dominate this use
    tl.store(src2dst_ptr + src_id, dst_id - num_invalid, mask=mask)
                                  ^
/home/coder/workspace/sglang-test/deepep_compute_src2dst_triton_kernel/test_deepep_compute_src2dst_triton_kernel.py:20:35: note: see current operation: %28 = "tensor.extract_slice"(%24, %14) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: -9223372036854775808>, static_strides = array<i64: 1>}> : (tensor<4xi32>, index) -> tensor<?xi32>
/home/coder/workspace/sglang-test/deepep_compute_src2dst_triton_kernel/test_deepep_compute_src2dst_triton_kernel.py:20:44: note: operand defined here (op in a parent region)
    tl.store(src2dst_ptr + src_id, dst_id - num_invalid, mask=mask)
                                           ^
<unknown>:0: warning: "linalg.yield"(%arg15) : (i32) -> () and its users all have no location!
<unknown>:0: note: see current operation: "linalg.yield"(%arg15) : (i32) -> ()
<unknown>:0: warning: "linalg.yield"(%arg11) : (i32) -> () and its users all have no location!
<unknown>:0: note: see current operation: "linalg.yield"(%arg11) : (i32) -> ()
///------------------[ERROR][Triton][END]------------------

[ERROR] 2025-07-16-02:49:13 (PID:41475, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
