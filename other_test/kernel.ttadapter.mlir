module {
  func.func @triton_load_store_sge_mask(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %0 = arith.muli %arg8, %c32768_i32 : i32
    scf.for %arg11 = %c0_i32 to %c32768_i32 step %c1024_i32  : i32 {
      %1 = arith.addi %0, %arg11 : i32
      %2 = arith.index_cast %1 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<1024xf32>
      %3 = arith.addi %2, %c1024 : index
      %4 = arith.index_cast %arg4 : i32 to index
      %5 = arith.maxsi %2, %4 : index
      %6 = arith.minsi %3, %5 : index
      %7 = arith.subi %6, %2 : index
      %8 = arith.subi %3, %6 : index
      %9 = arith.cmpi slt, %8, %c1024 : index
      scf.if %9 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
      }
      %subview = memref.subview %reinterpret_cast[%7] [%8] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %alloc[%7] [%8] [1] : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %10 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %extracted_slice = tensor.extract_slice %10[%7] [%8] [1] : tensor<1024xf32> to tensor<?xf32>
      %subview_2 = memref.subview %reinterpret_cast_1[%7] [%8] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

