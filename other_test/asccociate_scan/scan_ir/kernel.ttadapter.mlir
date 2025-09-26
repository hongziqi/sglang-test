#map = affine_map<(d0) -> (d0)>
module {
  func.func @scan_part_min_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi64> {tt.divisibility = 16 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0x7C00 : f16
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.muli %arg10, %c2_i32 : i32
    %1 = tensor.empty() : tensor<2xi32>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : tensor<2xi32>) {
    ^bb0(%out: i32):
      %34 = linalg.index 0 : index
      %35 = arith.index_cast %34 : index to i32
      linalg.yield %35 : i32
    } -> tensor<2xi32>
    %3 = linalg.fill ins(%0 : i32) outs(%1 : tensor<2xi32>) -> tensor<2xi32>
    %4 = arith.addi %3, %2 : tensor<2xi32>
    %5 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [2], strides: [1] : memref<?xf16> to memref<2xf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<2xf16>
    %6 = arith.addi %5, %c2 : index
    %7 = arith.index_cast %arg6 : i32 to index
    %8 = arith.maxsi %5, %7 : index
    %9 = arith.minsi %6, %8 : index
    %10 = arith.subi %9, %5 : index
    %11 = arith.cmpi slt, %10, %c2 : index
    scf.if %11 {
      linalg.fill ins(%cst : f16) outs(%alloc : memref<2xf16>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%10] [1] : memref<2xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%10] [1] : memref<2xf16> to memref<?xf16, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
    %12 = bufferization.to_tensor %alloc restrict writable : memref<2xf16>
    %13 = arith.extf %12 : tensor<2xf16> to tensor<2xf32>
    %alloc_1 = memref.alloc() : memref<2xf32>
    %alloc_2 = memref.alloc() : memref<2xi32>
    %extracted = tensor.extract %13[%c0] : tensor<2xf32>
    memref.store %extracted, %alloc_1[%c0] : memref<2xf32>
    %extracted_3 = tensor.extract %4[%c0] : tensor<2xi32>
    memref.store %extracted_3, %alloc_2[%c0] : memref<2xi32>
    %extracted_4 = tensor.extract %13[%c1] : tensor<2xf32>
    %14 = memref.load %alloc_1[%c0] : memref<2xf32>
    %extracted_5 = tensor.extract %4[%c1] : tensor<2xi32>
    %15 = memref.load %alloc_2[%c0] : memref<2xi32>
    %16 = arith.cmpf olt, %14, %extracted_4 : f32
    %17 = arith.cmpf oeq, %14, %extracted_4 : f32
    %18 = arith.cmpf une, %14, %14 : f32
    %19 = arith.cmpf une, %extracted_4, %extracted_4 : f32
    %20 = arith.xori %19, %true : i1
    %21 = arith.andi %18, %20 : i1
    %22 = arith.ori %16, %21 : i1
    %23 = arith.andi %18, %19 : i1
    %24 = arith.ori %17, %23 : i1
    %25 = arith.cmpi sgt, %15, %extracted_5 : i32
    %26 = arith.andi %24, %25 : i1
    %27 = arith.ori %22, %26 : i1
    %28 = arith.select %27, %14, %extracted_4 : f32
    %29 = arith.select %27, %15, %extracted_5 : i32
    memref.store %28, %alloc_1[%c1] : memref<2xf32>
    memref.store %29, %alloc_2[%c1] : memref<2xi32>
    %30 = bufferization.to_tensor %alloc_1 restrict : memref<2xf32>
    %31 = bufferization.to_tensor %alloc_2 restrict : memref<2xi32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%5], sizes: [2], strides: [1] : memref<?xf16> to memref<2xf16, strided<[1], offset: ?>>
    %32 = arith.truncf %30 : tensor<2xf32> to tensor<2xf16>
    %extracted_slice = tensor.extract_slice %32[0] [%10] [1] : tensor<2xf16> to tensor<?xf16>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%10] [1] : memref<2xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    %reinterpret_cast_8 = memref.reinterpret_cast %arg5 to offset: [%5], sizes: [2], strides: [1] : memref<?xi64> to memref<2xi64, strided<[1], offset: ?>>
    %33 = arith.extsi %31 : tensor<2xi32> to tensor<2xi64>
    %extracted_slice_9 = tensor.extract_slice %33[0] [%10] [1] : tensor<2xi64> to tensor<?xi64>
    %subview_10 = memref.subview %reinterpret_cast_8[0] [%10] [1] : memref<2xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice_9 in writable %subview_10 : (tensor<?xi64>, memref<?xi64, strided<[1], offset: ?>>) -> ()
    return
  }
}

