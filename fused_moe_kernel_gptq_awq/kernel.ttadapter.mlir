#map = affine_map<(d0) -> (d0)>
module {
  func.func @fused_moe_kernel_gptq_awq(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix"} {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 1.280000e+02 : f32
    %c15_i32 = arith.constant 15 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64
    %0 = tensor.empty() : tensor<16x16xi64>
    %1 = linalg.fill ins(%c16_i64 : i64) outs(%0 : tensor<16x16xi64>) -> tensor<16x16xi64>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = tensor.empty() : tensor<16x1xi64>
    %6 = linalg.fill ins(%c2_i64 : i64) outs(%5 : tensor<16x1xi64>) -> tensor<16x1xi64>
    %7 = tensor.empty() : tensor<16xi64>
    %8 = linalg.fill ins(%c32_i64 : i64) outs(%7 : tensor<16xi64>) -> tensor<16xi64>
    %9 = arith.addi %arg10, %c15_i32 : i32
    %10 = arith.divsi %9, %c16_i32 : i32
    %11 = arith.divsi %arg24, %c2_i32 : i32
    %12 = arith.subi %10, %11 : i32
    %13 = arith.minsi %12, %c1_i32 : i32
    %14 = arith.remsi %arg24, %c2_i32 : i32
    %15 = arith.remsi %14, %13 : i32
    %16 = arith.addi %11, %15 : i32
    %17 = arith.divsi %14, %13 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
    %18 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1]>>
    %19 = arith.muli %16, %c16_i32 : i32
    %20 = arith.cmpi sge, %19, %18 : i32
    scf.if %20 {
    } else {
      %21 = tensor.empty() : tensor<16xi32>
      %22 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%21 : tensor<16xi32>) {
      ^bb0(%out: i32):
        %60 = linalg.index 0 : index
        %61 = arith.index_cast %60 : index to i32
        linalg.yield %61 : i32
      } -> tensor<16xi32>
      %23 = arith.extsi %22 : tensor<16xi32> to tensor<16xi64>
      %24 = arith.index_cast %19 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%24], sizes: [16], strides: [1] : memref<?xi64> to memref<16xi64, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<16xi64>
      memref.copy %reinterpret_cast_1, %alloc : memref<16xi64, strided<[1], offset: ?>> to memref<16xi64>
      %25 = bufferization.to_tensor %alloc restrict writable : memref<16xi64>
      %alloc_2 = memref.alloc() : memref<16xi64>
      memref.copy %reinterpret_cast_1, %alloc_2 : memref<16xi64, strided<[1], offset: ?>> to memref<16xi64>
      %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<16xi64>
      %27 = arith.extsi %arg11 : i32 to i64
      %28 = linalg.fill ins(%27 : i64) outs(%7 : tensor<16xi64>) -> tensor<16xi64>
      %29 = arith.cmpi slt, %26, %28 : tensor<16xi64>
      %30 = arith.index_cast %16 : i32 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [%30], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %31 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>>
      %32 = arith.muli %17, %c16_i32 : i32
      %33 = arith.extsi %32 : i32 to i64
      %34 = linalg.fill ins(%33 : i64) outs(%7 : tensor<16xi64>) -> tensor<16xi64>
      %35 = arith.addi %34, %23 : tensor<16xi64>
      %36 = arith.remsi %35, %8 : tensor<16xi64>
      %expanded = tensor.expand_shape %25 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
      %37 = arith.divsi %expanded, %6 : tensor<16x1xi64>
      %38 = arith.extsi %arg12 : i32 to i64
      %39 = linalg.fill ins(%38 : i64) outs(%5 : tensor<16x1xi64>) -> tensor<16x1xi64>
      %40 = arith.muli %37, %39 : tensor<16x1xi64>
      %expanded_4 = tensor.expand_shape %22 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
      %41 = arith.extsi %expanded_4 : tensor<1x16xi32> to tensor<1x16xi64>
      %collapsed = tensor.collapse_shape %40 [[0, 1]] : tensor<16x1xi64> into tensor<16xi64>
      %broadcasted = linalg.broadcast ins(%collapsed : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [1] 
      %collapsed_5 = tensor.collapse_shape %41 [[0, 1]] : tensor<1x16xi64> into tensor<16xi64>
      %broadcasted_6 = linalg.broadcast ins(%collapsed_5 : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [0] 
      %42 = arith.addi %broadcasted, %broadcasted_6 : tensor<16x16xi64>
      %43 = arith.extsi %arg13 : i32 to i64
      %44 = arith.muli %31, %43 : i64
      %expanded_7 = tensor.expand_shape %22 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
      %45 = linalg.fill ins(%44 : i64) outs(%5 : tensor<16x1xi64>) -> tensor<16x1xi64>
      %46 = arith.extsi %expanded_7 : tensor<16x1xi32> to tensor<16x1xi64>
      %47 = arith.addi %45, %46 : tensor<16x1xi64>
      %expanded_8 = tensor.expand_shape %36 [[0, 1]] output_shape [1, 16] : tensor<16xi64> into tensor<1x16xi64>
      %48 = arith.extsi %arg14 : i32 to i64
      %49 = tensor.empty() : tensor<1x16xi64>
      %50 = linalg.fill ins(%48 : i64) outs(%49 : tensor<1x16xi64>) -> tensor<1x16xi64>
      %51 = arith.muli %expanded_8, %50 : tensor<1x16xi64>
      %collapsed_9 = tensor.collapse_shape %47 [[0, 1]] : tensor<16x1xi64> into tensor<16xi64>
      %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [1] 
      %collapsed_11 = tensor.collapse_shape %51 [[0, 1]] : tensor<1x16xi64> into tensor<16xi64>
      %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [0] 
      %52 = arith.addi %broadcasted_10, %broadcasted_12 : tensor<16x16xi64>
      %53 = tensor.empty() : tensor<16x16xi1>
      %broadcasted_13 = linalg.broadcast ins(%29 : tensor<16xi1>) outs(%53 : tensor<16x16xi1>) dimensions = [1] 
      %54 = arith.extsi %arg16 : i32 to i64
      %55 = arith.muli %31, %54 : i64
      %56 = arith.extsi %arg17 : i32 to i64
      %57:3 = scf.for %arg27 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg28 = %4, %arg29 = %42, %arg30 = %52) -> (tensor<16x16xf32>, tensor<16x16xi64>, tensor<16x16xi64>)  : i32 {
        %60 = arith.muli %arg27, %c16_i32 : i32
        %61 = arith.subi %c64_i32, %60 : i32
        %62 = tensor.empty() : tensor<1x16xi32>
        %63 = linalg.fill ins(%61 : i32) outs(%62 : tensor<1x16xi32>) -> tensor<1x16xi32>
        %64 = arith.cmpi slt, %expanded_4, %63 : tensor<1x16xi32>
        %collapsed_14 = tensor.collapse_shape %64 [[0, 1]] : tensor<1x16xi1> into tensor<16xi1>
        %broadcasted_15 = linalg.broadcast ins(%collapsed_14 : tensor<16xi1>) outs(%53 : tensor<16x16xi1>) dimensions = [0] 
        %65 = arith.andi %broadcasted_13, %broadcasted_15 : tensor<16x16xi1>
        %66 = scf.for %arg31 = %c0 to %c16 step %c1 iter_args(%arg32 = %2) -> (tensor<16x16xf32>) {
          %77 = scf.for %arg33 = %c0 to %c16 step %c1 iter_args(%arg34 = %arg32) -> (tensor<16x16xf32>) {
            %extracted = tensor.extract %arg29[%arg31, %arg33] {DiscreteMemAccess} : tensor<16x16xi64>
            %78 = arith.index_cast %extracted : i64 to index
            %reinterpret_cast_16 = memref.reinterpret_cast %arg2 to offset: [%78], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %79 = memref.load %reinterpret_cast_16[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %80 = tensor.empty() : tensor<1x1xf32>
            %81 = linalg.fill ins(%79 : f32) outs(%80 : tensor<1x1xf32>) -> tensor<1x1xf32>
            %inserted_slice = tensor.insert_slice %81 into %arg34[%arg31, %arg33] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
            scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
          } {ExtractedLoadOrStore}
          scf.yield %77 : tensor<16x16xf32>
        } {ExtractedLoadOrStore}
        %67 = arith.select %65, %66, %4 : tensor<16x16xi1>, tensor<16x16xf32>
        %68 = tensor.empty() : tensor<16x16xi8>
        %69 = scf.for %arg31 = %c0 to %c16 step %c1 iter_args(%arg32 = %68) -> (tensor<16x16xi8>) {
          %77 = scf.for %arg33 = %c0 to %c16 step %c1 iter_args(%arg34 = %arg32) -> (tensor<16x16xi8>) {
            %extracted = tensor.extract %arg30[%arg31, %arg33] {DiscreteMemAccess} : tensor<16x16xi64>
            %78 = arith.index_cast %extracted : i64 to index
            %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%78], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
            %79 = memref.load %reinterpret_cast_16[%c0] : memref<1xi8, strided<[1], offset: ?>>
            %80 = tensor.empty() : tensor<1x1xi8>
            %81 = linalg.fill ins(%79 : i8) outs(%80 : tensor<1x1xi8>) -> tensor<1x1xi8>
            %inserted_slice = tensor.insert_slice %81 into %arg34[%arg31, %arg33] [1, 1] [16, 1] : tensor<1x1xi8> into tensor<16x16xi8>
            scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xi8>
          } {ExtractedLoadOrStore}
          scf.yield %77 : tensor<16x16xi8>
        } {ExtractedLoadOrStore}
        %70 = scf.for %arg31 = %c0 to %c16 step %c1 iter_args(%arg32 = %2) -> (tensor<16x16xf32>) {
          %77 = scf.for %arg33 = %c0 to %c16 step %c1 iter_args(%arg34 = %arg32) -> (tensor<16x16xf32>) {
            %78 = arith.index_cast %arg33 : index to i32
            %79 = arith.extsi %78 : i32 to i64
            %80 = arith.addi %33, %79 : i64
            %81 = arith.remsi %80, %c32_i64 : i64
            %82 = arith.muli %81, %56 : i64
            %83 = arith.addi %55, %82 : i64
            %84 = arith.index_cast %arg31 : index to i32
            %85 = arith.addi %84, %60 : i32
            %86 = arith.divsi %85, %c16_i32 : i32
            %87 = arith.extsi %86 : i32 to i64
            %88 = arith.addi %83, %87 : i64
            %89 = arith.index_cast %88 : i64 to index
            %reinterpret_cast_16 = memref.reinterpret_cast %arg5 to offset: [%89], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %90 = memref.load %reinterpret_cast_16[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %91 = tensor.empty() : tensor<1x1xf32>
            %92 = linalg.fill ins(%90 : f32) outs(%91 : tensor<1x1xf32>) -> tensor<1x1xf32>
            %inserted_slice = tensor.insert_slice %92 into %arg34[%arg31, %arg33] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
            scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
          } {ExtractedLoadOrStore}
          scf.yield %77 : tensor<16x16xf32>
        } {ExtractedLoadOrStore}
        %71 = arith.sitofp %69 : tensor<16x16xi8> to tensor<16x16xf32>
        %72 = arith.subf %71, %3 : tensor<16x16xf32>
        %73 = arith.mulf %72, %70 : tensor<16x16xf32>
        %74 = linalg.matmul {input_precison = "ieee"} ins(%67, %73 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%arg28 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %75 = arith.addi %arg29, %1 : tensor<16x16xi64>
        %76 = arith.addi %arg30, %1 : tensor<16x16xi64>
        scf.yield %74, %75, %76 : tensor<16x16xf32>, tensor<16x16xi64>, tensor<16x16xi64>
      }
      %58 = arith.extsi %arg15 : i32 to i64
      %59 = scf.for %arg27 = %c0 to %c16 step %c1 iter_args(%arg28 = %2) -> (tensor<16x16xf32>) {
        %60 = scf.for %arg29 = %c0 to %c16 step %c1 iter_args(%arg30 = %arg28) -> (tensor<16x16xf32>) {
          %extracted = tensor.extract %25[%arg27] {DiscreteMemAccess} : tensor<16xi64>
          %61 = arith.muli %58, %extracted : i64
          %62 = arith.index_cast %arg29 : index to i32
          %63 = arith.addi %32, %62 : i32
          %64 = arith.extsi %63 : i32 to i64
          %65 = arith.addi %61, %64 : i64
          %66 = arith.index_cast %65 : i64 to index
          %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%66], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          %67 = memref.load %reinterpret_cast_14[%c0] : memref<1xf32, strided<[1], offset: ?>>
          %68 = tensor.empty() : tensor<1x1xf32>
          %69 = linalg.fill ins(%67 : f32) outs(%68 : tensor<1x1xf32>) -> tensor<1x1xf32>
          %inserted_slice = tensor.insert_slice %69 into %arg30[%arg27, %arg29] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
          scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
        } {ExtractedLoadOrStore}
        scf.yield %60 : tensor<16x16xf32>
      } {ExtractedLoadOrStore}
      scf.for %arg27 = %c0 to %c16 step %c1 {
        scf.for %arg28 = %c0 to %c16 step %c1 {
          %extracted = tensor.extract %26[%arg27] {DiscreteMemAccess} : tensor<16xi64>
          %60 = arith.muli %58, %extracted : i64
          %61 = arith.index_cast %arg28 : index to i32
          %62 = arith.addi %32, %61 : i32
          %63 = arith.extsi %62 : i32 to i64
          %64 = arith.addi %60, %63 : i64
          %65 = arith.index_cast %64 : i64 to index
          %66 = arith.cmpi slt, %extracted, %27 : i64
          %67 = arith.cmpi slt, %62, %c32_i32 : i32
          %68 = arith.andi %66, %67 : i1
          %extracted_14 = tensor.extract %57#0[%arg27, %arg28] {DiscreteMemAccess} : tensor<16x16xf32>
          %extracted_15 = tensor.extract %59[%arg27, %arg28] {DiscreteMemAccess} : tensor<16x16xf32>
          %69 = arith.select %68, %extracted_14, %extracted_15 : f32
          %70 = tensor.empty() : tensor<1xf32>
          %71 = linalg.fill ins(%69 : f32) outs(%70 : tensor<1xf32>) -> tensor<1xf32>
          %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%65], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          bufferization.materialize_in_destination %71 in writable %reinterpret_cast_16 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
        } {ExtractedLoadOrStore}
      } {ExtractedLoadOrStore}
    }
    return
  }
}

