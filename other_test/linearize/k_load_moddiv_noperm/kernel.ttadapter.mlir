#map = affine_map<(d0) -> (d0)>
module {
  func.func @k_load_moddiv_noperm(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8192 = arith.constant 8192 : index
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant dense<[64, 4]> : tensor<2xi64>
    %c2048 = arith.constant 2048 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x4xf32>) -> tensor<64x4xf32>
    %2 = arith.muli %arg10, %c64_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = tensor.empty() : tensor<64xi32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %29 = linalg.index 0 : index
      %30 = arith.index_cast %29 : index to i32
      linalg.yield %30 : i32
    } -> tensor<64xi32>
    %6 = linalg.fill ins(%2 : i32) outs(%4 : tensor<64xi32>) -> tensor<64xi32>
    %7 = arith.addi %6, %5 : tensor<64xi32>
    %8 = arith.muli %arg9, %c4_i32 : i32
    %expanded = tensor.expand_shape %7 [[0, 1]] output_shape [64, 1] : tensor<64xi32> into tensor<64x1xi32>
    %9 = tensor.empty() : tensor<64x1xi32>
    %10 = linalg.fill ins(%arg4 : i32) outs(%9 : tensor<64x1xi32>) -> tensor<64x1xi32>
    %11 = arith.cmpi slt, %expanded, %10 : tensor<64x1xi32>
    %12 = arith.divsi %3, %c2048 : index
    %13 = arith.remsi %3, %c2048 : index
    %14 = arith.muli %12, %c8192 : index
    %15 = arith.muli %13, %c4 : index
    %16 = arith.index_cast %8 : i32 to index
    %17 = arith.addi %14, %15 : index
    %18 = arith.addi %17, %16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [1, 64, 4], strides: [256, 4, 1] : memref<?xf32> to memref<1x64x4xf32, strided<[256, 4, 1], offset: ?>>
    %19 = tensor.empty() : tensor<64x4xi1>
    %collapsed = tensor.collapse_shape %11 [[0, 1]] : tensor<64x1xi1> into tensor<64xi1>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xi1>) outs(%19 : tensor<64x4xi1>) dimensions = [1] 
    %alloc = memref.alloc() : memref<1x64x4xf32>
    memref.copy %reinterpret_cast, %alloc : memref<1x64x4xf32, strided<[256, 4, 1], offset: ?>> to memref<1x64x4xf32>
    %20 = bufferization.to_tensor %alloc restrict writable : memref<1x64x4xf32>
    %reshape = tensor.reshape %20(%cst) : (tensor<1x64x4xf32>, tensor<2xi64>) -> tensor<64x4xf32>
    %21 = arith.select %broadcasted, %reshape, %1 : tensor<64x4xi1>, tensor<64x4xf32>
    %22 = arith.muli %3, %c4 : index
    %23 = arith.addi %22, %16 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [64, 4], strides: [4, 1] : memref<?xf32> to memref<64x4xf32, strided<[4, 1], offset: ?>>
    %24 = arith.addi %3, %c64 : index
    %25 = arith.index_cast %arg4 : i32 to index
    %26 = arith.maxsi %3, %25 : index
    %27 = arith.minsi %24, %26 : index
    %28 = arith.subi %27, %3 : index
    %extracted_slice = tensor.extract_slice %21[0, 0] [%28, 4] [1, 1] : tensor<64x4xf32> to tensor<?x4xf32>
    %subview = memref.subview %reinterpret_cast_1[0, 0] [%28, 4] [1, 1] : memref<64x4xf32, strided<[4, 1], offset: ?>> to memref<?x4xf32, strided<[4, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x4xf32>, memref<?x4xf32, strided<[4, 1], offset: ?>>) -> ()
    return
  }
}

