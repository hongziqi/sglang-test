#map = affine_map<(d0) -> (d0)>
module {
  func.func @save_cache_to_buffer_with_mask(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32}, %arg6: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c10 = arith.constant 10 : index
    %c15_i32 = arith.constant 15 : i32
    %c5_i32 = arith.constant 5 : i32
    %c10_i32 = arith.constant 10 : i32
    %c150_i32 = arith.constant 150 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i8 = arith.constant 0 : i8
    %c5 = arith.constant 5 : index
    %c15_i64 = arith.constant 15 : i64
    %0 = tensor.empty() : tensor<1xi64>
    %1 = linalg.fill ins(%c15_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
    %2 = tensor.empty() : tensor<15xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<15xi8>) -> tensor<15xi8>
    %4 = tensor.empty() : tensor<15xi32>
    %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<15xi32>) -> tensor<15xi32>
    %6 = linalg.fill ins(%c1_i32 : i32) outs(%4 : tensor<15xi32>) -> tensor<15xi32>
    %7 = linalg.fill ins(%c150_i32 : i32) outs(%4 : tensor<15xi32>) -> tensor<15xi32>
    %8 = linalg.fill ins(%c10_i32 : i32) outs(%4 : tensor<15xi32>) -> tensor<15xi32>
    %9 = linalg.fill ins(%c5_i32 : i32) outs(%4 : tensor<15xi32>) -> tensor<15xi32>
    %10 = arith.muli %arg11, %c15_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<15xi32>) {
    ^bb0(%out: i32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i32
      linalg.yield %26 : i32
    } -> tensor<15xi32>
    %13 = linalg.fill ins(%10 : i32) outs(%4 : tensor<15xi32>) -> tensor<15xi32>
    %14 = arith.addi %13, %12 : tensor<15xi32>
    %15 = arith.divsi %14, %9 : tensor<15xi32>
    %16 = arith.remsi %14, %9 : tensor<15xi32>
    %17 = arith.muli %15, %8 : tensor<15xi32>
    %18 = arith.addi %17, %16 : tensor<15xi32>
    %19 = arith.cmpi slt, %18, %7 : tensor<15xi32>
    %20 = arith.select %19, %6, %5 : tensor<15xi1>, tensor<15xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%11], sizes: [15], strides: [1] : memref<?xi8> to memref<15xi8, strided<[1], offset: ?>>
    %21 = arith.trunci %20 : tensor<15xi32> to tensor<15xi8>
    bufferization.materialize_in_destination %21 in writable %reinterpret_cast : (tensor<15xi8>, memref<15xi8, strided<[1], offset: ?>>) -> ()
    scf.for %arg14 = %c0_i32 to %c15_i32 step %c1_i32  : i32 {
      %25 = arith.addi %10, %arg14 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = arith.divsi %25, %c5_i32 : i32
      %28 = arith.muli %27, %c10_i32 : i32
      %29 = arith.remsi %25, %c5_i32 : i32
      %30 = arith.addi %28, %29 : i32
      %31 = arith.trunci %30 : i32 to i8
      %32 = tensor.empty() : tensor<1xi8>
      %33 = linalg.fill ins(%31 : i8) outs(%32 : tensor<1xi8>) -> tensor<1xi8>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %33 in writable %reinterpret_cast_0 : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
    }
    %22 = arith.remsi %arg11, %c2_i32 : i32
    %23 = arith.cmpi eq, %22, %c0_i32 : i32
    scf.if %23 {
      %25 = arith.divsi %11, %c5 : index
      %26 = arith.remsi %11, %c5 : index
      %27 = arith.muli %25, %c10 : index
      %28 = arith.addi %27, %26 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [3, 5], strides: [10, 1] : memref<?xi8> to memref<3x5xi8, strided<[10, 1], offset: ?>>
      %alloc = memref.alloc() : memref<3x5xi8>
      memref.copy %reinterpret_cast_0, %alloc : memref<3x5xi8, strided<[10, 1], offset: ?>> to memref<3x5xi8>
      %29 = bufferization.to_tensor %alloc restrict writable : memref<3x5xi8>
      %reshape = tensor.reshape %29(%1) : (tensor<3x5xi8>, tensor<1xi64>) -> tensor<15xi8>
      %30 = arith.select %19, %reshape, %3 : tensor<15xi1>, tensor<15xi8>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [15], strides: [1] : memref<?xi8> to memref<15xi8, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %30 in writable %reinterpret_cast_1 : (tensor<15xi8>, memref<15xi8, strided<[1], offset: ?>>) -> ()
    }
    %24 = arith.cmpi eq, %22, %c1_i32 : i32
    scf.if %24 {
      %25 = arith.divsi %11, %c5 : index
      %26 = arith.remsi %11, %c5 : index
      %27 = arith.muli %25, %c10 : index
      %28 = arith.addi %27, %26 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%28], sizes: [3, 5], strides: [10, 1] : memref<?xi8> to memref<3x5xi8, strided<[10, 1], offset: ?>>
      %alloc = memref.alloc() : memref<3x5xi8>
      memref.copy %reinterpret_cast_0, %alloc : memref<3x5xi8, strided<[10, 1], offset: ?>> to memref<3x5xi8>
      %29 = bufferization.to_tensor %alloc restrict writable : memref<3x5xi8>
      %reshape = tensor.reshape %29(%1) : (tensor<3x5xi8>, tensor<1xi64>) -> tensor<15xi8>
      %30 = arith.select %19, %reshape, %3 : tensor<15xi1>, tensor<15xi8>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [15], strides: [1] : memref<?xi8> to memref<15xi8, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %30 in writable %reinterpret_cast_1 : (tensor<15xi8>, memref<15xi8, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

