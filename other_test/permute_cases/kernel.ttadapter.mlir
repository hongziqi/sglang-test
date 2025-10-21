#map = affine_map<(d0) -> (d0)>
module {
  func.func @triton_gpu(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %2 = tensor.empty() : tensor<4x1xi32>
    %3 = linalg.fill ins(%c4_i32 : i32) outs(%2 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %4 = arith.addi %arg12, %c1_i32 : i32
    %5 = arith.muli %arg11, %4 : i32
    %6 = arith.muli %5, %c256_i32 : i32
    %7 = arith.muli %arg10, %c4_i32 : i32
    %8 = tensor.empty() : tensor<4xi32>
    %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%8 : tensor<4xi32>) {
    ^bb0(%out: i32):
      %27 = linalg.index 0 : index
      %28 = arith.index_cast %27 : index to i32
      linalg.yield %28 : i32
    } -> tensor<4xi32>
    %expanded = tensor.expand_shape %9 [[0, 1]] output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
    %10 = linalg.fill ins(%7 : i32) outs(%2 : tensor<4x1xi32>) -> tensor<4x1xi32>
    %11 = arith.addi %10, %expanded : tensor<4x1xi32>
    %12 = arith.cmpi slt, %11, %3 : tensor<4x1xi32>
    %13 = arith.index_cast %7 : i32 to index
    %14 = arith.index_cast %6 : i32 to index
    %15 = arith.muli %14, %c4 : index
    %16 = arith.addi %13, %15 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [4, 256], strides: [1, 4] : memref<?xf32> to memref<4x256xf32, strided<[1, 4], offset: ?>>
    %17 = tensor.empty() : tensor<4x256xi1>
    %collapsed = tensor.collapse_shape %12 [[0, 1]] : tensor<4x1xi1> into tensor<4xi1>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<4xi1>) outs(%17 : tensor<4x256xi1>) dimensions = [1] 
    %alloc = memref.alloc() : memref<4x256xf32>
    %18 = arith.addi %13, %c4 : index
    %19 = arith.maxsi %13, %c4 : index
    %20 = arith.minsi %18, %19 : index
    %21 = arith.subi %20, %13 : index
    %22 = arith.cmpi slt, %21, %c4 : index
    scf.if %22 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<4x256xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0, 0] [%21, 256] [1, 1] : memref<4x256xf32, strided<[1, 4], offset: ?>> to memref<?x256xf32, strided<[1, 4], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0] [%21, 256] [1, 1] : memref<4x256xf32> to memref<?x256xf32, strided<[256, 1]>>
    memref.copy %subview, %subview_0 : memref<?x256xf32, strided<[1, 4], offset: ?>> to memref<?x256xf32, strided<[256, 1]>>
    %23 = bufferization.to_tensor %alloc restrict writable : memref<4x256xf32>
    %24 = scf.for %arg13 = %c0 to %c4 step %c1 iter_args(%arg14 = %0) -> (tensor<4x256xf32>) {
      %27 = scf.for %arg15 = %c0 to %c256 step %c1 iter_args(%arg16 = %arg14) -> (tensor<4x256xf32>) {
        %28 = arith.index_cast %arg15 : index to i32
        %29 = arith.addi %6, %28 : i32
        %30 = arith.remsi %29, %c2048_i32 : i32
        %31 = arith.index_cast %arg13 : index to i32
        %32 = arith.addi %7, %31 : i32
        %33 = arith.muli %32, %c2048_i32 : i32
        %34 = arith.addi %30, %33 : i32
        %35 = arith.divsi %29, %c2048_i32 : i32
        %36 = arith.muli %35, %c8192_i32 : i32
        %37 = arith.addi %34, %36 : i32
        %38 = arith.index_cast %37 : i32 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
        %alloc_4 = memref.alloc() : memref<1xf32>
        memref.copy %reinterpret_cast_3, %alloc_4 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32>
        %39 = bufferization.to_tensor %alloc_4 restrict writable : memref<1xf32>
        %extracted = tensor.extract %39[%c0] : tensor<1xf32>
        %40 = tensor.empty() : tensor<1x1xf32>
        %41 = linalg.fill ins(%extracted : f32) outs(%40 : tensor<1x1xf32>) -> tensor<1x1xf32>
        %inserted_slice = tensor.insert_slice %41 into %arg16[%arg13, %arg15] [1, 1] [256, 1] : tensor<1x1xf32> into tensor<4x256xf32>
        scf.yield {DiscreteMemAccess} %inserted_slice : tensor<4x256xf32>
      } {ExtractedLoadOrStore}
      scf.yield %27 : tensor<4x256xf32>
    } {ExtractedLoadOrStore}
    %25 = arith.select %broadcasted, %24, %1 {DiscreteMemAccess} : tensor<4x256xi1>, tensor<4x256xf32>
    %26 = arith.addf %23, %25 : tensor<4x256xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [4, 256], strides: [1, 4] : memref<?xf32> to memref<4x256xf32, strided<[1, 4], offset: ?>>
    %extracted_slice = tensor.extract_slice %26[0, 0] [%21, 256] [1, 1] : tensor<4x256xf32> to tensor<?x256xf32>
    %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%21, 256] [1, 1] : memref<4x256xf32, strided<[1, 4], offset: ?>> to memref<?x256xf32, strided<[1, 4], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_2 : (tensor<?x256xf32>, memref<?x256xf32, strided<[1, 4], offset: ?>>) -> ()
    return
  }
}

