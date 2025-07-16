(triton) (triton) coder@candy-npu:~/workspace/sglang-test/_per_token_group_quant_8bit$ python test_per_token_group_quant_8bit.py 
Traceback (most recent call last):
  File "/home/coder/workspace/sglang-test/_per_token_group_quant_8bit/test_per_token_group_quant_8bit.py", line 143, in <module>
    save_inputs_outputs(path, dst_type=fp8_type_)
  File "/home/coder/workspace/sglang-test/_per_token_group_quant_8bit/test_per_token_group_quant_8bit.py", line 102, in save_inputs_outputs
    y_q, y_s = triton_per_token_group_quant_8bit(x, group_size, dst_type, eps, BLOCK_SIZE)
  File "/home/coder/workspace/sglang-test/_per_token_group_quant_8bit/test_per_token_group_quant_8bit.py", line 81, in triton_per_token_group_quant_8bit
    _per_token_group_quant_8bit[grid](
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/runtime/jit.py", line 331, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/runtime/jit.py", line 625, in run
    kernel = self.compile(
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/compiler/compiler.py", line 281, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
  File "/home/coder/miniconda/envs/triton/lib/python3.10/site-packages/triton/compiler/compiler.py", line 102, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
triton.compiler.errors.CompilationError: at 1:0:
def _per_token_group_quant_8bit(
^
AttributeError("'NPUOptions' object has no attribute 'supported_fp8_dtypes'")
[ERROR] 2025-07-16-09:10:29 (PID:214287, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception