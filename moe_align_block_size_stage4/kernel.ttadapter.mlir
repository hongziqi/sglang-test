module {
  func.func @moe_align_block_size_stage4(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %1 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
    %2 = arith.addi %0, %c1 : index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%2], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %3 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>>
    scf.for %arg13 = %1 to %3 step %c2_i32  : i32 {
      %8 = arith.divsi %arg13, %c2_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = tensor.empty() : tensor<1xi32>
      %11 = linalg.fill ins(%arg10 : i32) outs(%10 : tensor<1xi32>) -> tensor<1xi32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %11 in writable %reinterpret_cast_1 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
    }
    %4 = arith.muli %arg10, %c2_i32 : i32   // token_start = pid * tokens_per_thread（2） = 0 * 2 = 0
    %5 = arith.addi %4, %c2_i32 : i32       // token_start + tokens_per_thread（临时end）
    %6 = arith.minsi %5, %c4_i32 : i32      // token_end = min(临时end, numel=4)
    %7 = arith.index_cast %4 : i32 to index // token_start 转 index（用于tokens_cnts地址计算）

    // 遍历每个Token i（步长=1） 线程0: toekn[0,2)
    scf.for %arg13 = %4 to %6 step %c1_i32  : i32 {
      // 1. 加载topk_ids[i] → expert_id（%9）
      %8 = arith.index_cast %arg13 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %9 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>>

      // 2. 加载tokens_cnts[off_t + expert_id] → token_cnt（%12）
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.addi %7, %10 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%11], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %12 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>

      // 3. 加载cumsum[expert_id]，计算rank_post_pad（%14）
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %13 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
      %14 = arith.addi %12, %13 : i32

      // 4. 存储i到sorted_token_ids[rank_post_pad]
      %15 = arith.index_cast %14 : i32 to index
      %16 = tensor.empty() : tensor<1xi32>
      %17 = linalg.fill ins(%arg13 : i32) outs(%16 : tensor<1xi32>) -> tensor<1xi32>
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %17 in writable %reinterpret_cast_4 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
      
      // 5. 更新tokens_cnts[off_t + expert_id] = token_cnt + 1
      %18 = arith.addi %12, %c1_i32 : i32
      %19 = linalg.fill ins(%18 : i32) outs(%16 : tensor<1xi32>) -> tensor<1xi32>
      bufferization.materialize_in_destination %19 in writable %reinterpret_cast_2 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

