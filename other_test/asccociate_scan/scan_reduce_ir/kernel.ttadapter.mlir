module {
  func.func @scan_part_min_kernel_bak(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi64> {tt.divisibility = 16 : i32}, %arg5: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0x7C00 : f16
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.muli %arg12, %c2_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [2], strides: [1] : memref<?xf16> to memref<2xf16, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<2xf16>
    %2 = arith.addi %1, %c2 : index
    %3 = arith.index_cast %arg8 : i32 to index
    %4 = arith.maxsi %1, %3 : index
    %5 = arith.minsi %2, %4 : index
    %6 = arith.subi %5, %1 : index
    %7 = arith.cmpi slt, %6, %c2 : index
    scf.if %7 {
      linalg.fill ins(%cst : f16) outs(%alloc : memref<2xf16>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<2xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<2xf16> to memref<?xf16, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<2xf16>
    %9 = arith.extf %8 : tensor<2xf16> to tensor<2xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%1], sizes: [2], strides: [1] : memref<?xi64> to memref<2xi64, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<2xi64>
    scf.if %7 {
      linalg.fill ins(%c0_i64 : i64) outs(%alloc_2 : memref<2xi64>)
    }
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<2xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<2xi64> to memref<?xi64, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xi64, strided<[1], offset: ?>> to memref<?xi64, strided<[1]>>
    %10 = bufferization.to_tensor %alloc_2 restrict writable : memref<2xi64>
    %alloc_5 = memref.alloc() : memref<2xf32>
    %alloc_6 = memref.alloc() : memref<2xi64>
    %extracted = tensor.extract %9[%c0] : tensor<2xf32>
    memref.store %extracted, %alloc_5[%c0] : memref<2xf32>
    %11 = memref.load %alloc_2[%c0] : memref<2xi64>
    memref.store %11, %alloc_6[%c0] : memref<2xi64>
    %extracted_7 = tensor.extract %9[%c1] : tensor<2xf32>
    %12 = memref.load %alloc_5[%c0] : memref<2xf32>
    %13 = memref.load %alloc_2[%c1] : memref<2xi64>
    %14 = memref.load %alloc_6[%c0] : memref<2xi64>
    %15 = arith.cmpf olt, %12, %extracted_7 : f32
    %16 = arith.cmpf oeq, %12, %extracted_7 : f32
    %17 = arith.cmpf une, %12, %12 : f32
    %18 = arith.cmpf une, %extracted_7, %extracted_7 : f32
    %19 = arith.xori %18, %true : i1
    %20 = arith.andi %17, %19 : i1
    %21 = arith.ori %15, %20 : i1
    %22 = arith.andi %17, %18 : i1
    %23 = arith.ori %16, %22 : i1
    %24 = arith.cmpi sgt, %14, %13 : i64
    %25 = arith.andi %23, %24 : i1
    %26 = arith.ori %21, %25 : i1
    %27 = arith.select %26, %12, %extracted_7 : f32
    %28 = arith.select %26, %14, %13 : i64
    memref.store %27, %alloc_5[%c1] : memref<2xf32>
    memref.store %28, %alloc_6[%c1] : memref<2xi64>
    %29 = bufferization.to_tensor %alloc_5 restrict : memref<2xf32>
    %30 = bufferization.to_tensor %alloc_6 restrict : memref<2xi64>
    %31 = tensor.empty() : tensor<f32>
    %32 = tensor.empty() : tensor<i64>
    %reduced:2 = linalg.reduce ins(%9, %10 : tensor<2xf32>, tensor<2xi64>) outs(%31, %32 : tensor<f32>, tensor<i64>) dimensions = [0]  {reduce_mode = "min_with_index"}
      (%in: f32, %in_15: i64, %init: f32, %init_16: i64) {
        %39 = arith.cmpf olt, %in, %init : f32
        %40 = arith.cmpf oeq, %in, %init : f32
        %41 = arith.cmpf une, %in, %in : f32
        %42 = arith.cmpf une, %init, %init : f32
        %43 = arith.xori %42, %true : i1
        %44 = arith.andi %41, %43 : i1
        %45 = arith.ori %39, %44 : i1
        %46 = arith.andi %41, %42 : i1
        %47 = arith.ori %40, %46 : i1
        %48 = arith.cmpi sgt, %in_15, %init_16 : i64
        %49 = arith.andi %47, %48 : i1
        %50 = arith.ori %45, %49 : i1
        %51 = arith.select %50, %in, %init : f32
        %52 = arith.select %50, %in_15, %init_16 : i64
        linalg.yield %51, %52 : f32, i64
      }
    %extracted_8 = tensor.extract %reduced#0[] : tensor<f32>
    %extracted_9 = tensor.extract %reduced#1[] : tensor<i64>
    %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [2], strides: [1] : memref<?xf16> to memref<2xf16, strided<[1], offset: ?>>
    %33 = arith.truncf %29 : tensor<2xf32> to tensor<2xf16>
    %extracted_slice = tensor.extract_slice %33[0] [%6] [1] : tensor<2xf16> to tensor<?xf16>
    %subview_11 = memref.subview %reinterpret_cast_10[0] [%6] [1] : memref<2xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_11 : (tensor<?xf16>, memref<?xf16, strided<[1], offset: ?>>) -> ()
    %extracted_slice_12 = tensor.extract_slice %30[0] [%6] [1] : tensor<2xi64> to tensor<?xi64>
    bufferization.materialize_in_destination %extracted_slice_12 in writable %subview_3 : (tensor<?xi64>, memref<?xi64, strided<[1], offset: ?>>) -> ()
    %34 = arith.index_cast %arg12 : i32 to index
    %35 = tensor.empty() : tensor<1xf32>
    %36 = linalg.fill ins(%extracted_8 : f32) outs(%35 : tensor<1xf32>) -> tensor<1xf32>
    %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%34], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %36 in writable %reinterpret_cast_13 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
    %37 = tensor.empty() : tensor<1xi64>
    %38 = linalg.fill ins(%extracted_9 : i64) outs(%37 : tensor<1xi64>) -> tensor<1xi64>
    %reinterpret_cast_14 = memref.reinterpret_cast %arg7 to offset: [%34], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %38 in writable %reinterpret_cast_14 : (tensor<1xi64>, memref<1xi64, strided<[1], offset: ?>>) -> ()
    return
  }
}

