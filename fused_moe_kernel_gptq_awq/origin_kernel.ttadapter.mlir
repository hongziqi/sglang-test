#map = affine_map<(d0) -> (d0)>
module {
  func.func @fused_moe_kernel_gptq_awq(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32}, %arg7: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32, %arg25: i32, %arg26: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix"} {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i64 = arith.constant 2 : i64
    %c64_i32 = arith.constant 64 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.280000e+02 : f32
    %c15_i32 = arith.constant 15 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32_i64 = arith.constant 32 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c16_i64 = arith.constant 16 : i64
    %0 = tensor.empty() : tensor<16x16xi64>
    %1 = linalg.fill ins(%c16_i64 : i64) outs(%0 : tensor<16x16xi64>) -> tensor<16x16xi64>
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
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
        %33 = linalg.index 0 : index
        %34 = arith.index_cast %33 : index to i32
        linalg.yield %34 : i32
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
      %32 = arith.cmpi eq, %31, %c-1_i64 : i64
      scf.if %32 {
        %33 = arith.muli %17, %c16_i32 : i32
        %34 = arith.extsi %arg15 : i32 to i64
        %35 = scf.for %arg27 = %c0 to %c16 step %c1 iter_args(%arg28 = %2) -> (tensor<16x16xf32>) {
          %36 = scf.for %arg29 = %c0 to %c16 step %c1 iter_args(%arg30 = %arg28) -> (tensor<16x16xf32>) {
            %extracted = tensor.extract %25[%arg27] {DiscreteMemAccess} : tensor<16xi64>
            %37 = arith.muli %34, %extracted : i64
            %38 = arith.index_cast %arg29 : index to i32
            %39 = arith.addi %33, %38 : i32
            %40 = arith.extsi %39 : i32 to i64
            %41 = arith.addi %37, %40 : i64
            %42 = arith.index_cast %41 : i64 to index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %43 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %44 = tensor.empty() : tensor<1x1xf32>
            %45 = linalg.fill ins(%43 : f32) outs(%44 : tensor<1x1xf32>) -> tensor<1x1xf32>
            %inserted_slice = tensor.insert_slice %45 into %arg30[%arg27, %arg29] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
            scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
          } {ExtractedLoadOrStore}
          scf.yield %36 : tensor<16x16xf32>
        } {ExtractedLoadOrStore}
        scf.for %arg27 = %c0 to %c16 step %c1 {
          scf.for %arg28 = %c0 to %c16 step %c1 {
            %extracted = tensor.extract %26[%arg27] {DiscreteMemAccess} : tensor<16xi64>
            %36 = arith.muli %34, %extracted : i64
            %37 = arith.index_cast %arg28 : index to i32
            %38 = arith.addi %33, %37 : i32
            %39 = arith.extsi %38 : i32 to i64
            %40 = arith.addi %36, %39 : i64
            %41 = arith.index_cast %40 : i64 to index
            %42 = arith.cmpi slt, %extracted, %27 : i64
            %43 = arith.cmpi slt, %38, %c32_i32 : i32
            %44 = arith.andi %42, %43 : i1
            %extracted_4 = tensor.extract %35[%arg27, %arg28] {DiscreteMemAccess} : tensor<16x16xf32>
            %45 = arith.select %44, %cst_0, %extracted_4 : f32
            %46 = tensor.empty() : tensor<1xf32>
            %47 = linalg.fill ins(%45 : f32) outs(%46 : tensor<1xf32>) -> tensor<1xf32>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            bufferization.materialize_in_destination %47 in writable %reinterpret_cast_5 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
          } {ExtractedLoadOrStore}
        } {ExtractedLoadOrStore}
      } else {
        %33 = arith.muli %17, %c16_i32 : i32
        %34 = arith.extsi %33 : i32 to i64
        %35 = linalg.fill ins(%34 : i64) outs(%7 : tensor<16xi64>) -> tensor<16xi64>
        %36 = arith.addi %35, %23 : tensor<16xi64>
        %37 = arith.remsi %36, %8 : tensor<16xi64>
        %expanded = tensor.expand_shape %25 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
        %38 = arith.divsi %expanded, %6 : tensor<16x1xi64>
        %39 = arith.extsi %arg12 : i32 to i64
        %40 = linalg.fill ins(%39 : i64) outs(%5 : tensor<16x1xi64>) -> tensor<16x1xi64>
        %41 = arith.muli %38, %40 : tensor<16x1xi64>
        %expanded_4 = tensor.expand_shape %22 [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
        %42 = arith.extsi %expanded_4 : tensor<1x16xi32> to tensor<1x16xi64>
        %collapsed = tensor.collapse_shape %41 [[0, 1]] : tensor<16x1xi64> into tensor<16xi64>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [1] 
        %collapsed_5 = tensor.collapse_shape %42 [[0, 1]] : tensor<1x16xi64> into tensor<16xi64>
        %broadcasted_6 = linalg.broadcast ins(%collapsed_5 : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [0] 
        %43 = arith.addi %broadcasted, %broadcasted_6 : tensor<16x16xi64>
        %44 = arith.extsi %arg13 : i32 to i64
        %45 = arith.muli %31, %44 : i64
        %expanded_7 = tensor.expand_shape %22 [[0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
        %46 = linalg.fill ins(%45 : i64) outs(%5 : tensor<16x1xi64>) -> tensor<16x1xi64>
        %47 = arith.extsi %expanded_7 : tensor<16x1xi32> to tensor<16x1xi64>
        %48 = arith.addi %46, %47 : tensor<16x1xi64>
        %expanded_8 = tensor.expand_shape %37 [[0, 1]] output_shape [1, 16] : tensor<16xi64> into tensor<1x16xi64>
        %49 = arith.extsi %arg14 : i32 to i64
        %50 = tensor.empty() : tensor<1x16xi64>
        %51 = linalg.fill ins(%49 : i64) outs(%50 : tensor<1x16xi64>) -> tensor<1x16xi64>
        %52 = arith.muli %expanded_8, %51 : tensor<1x16xi64>
        %collapsed_9 = tensor.collapse_shape %48 [[0, 1]] : tensor<16x1xi64> into tensor<16xi64>
        %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [1] 
        %collapsed_11 = tensor.collapse_shape %52 [[0, 1]] : tensor<1x16xi64> into tensor<16xi64>
        %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<16xi64>) outs(%0 : tensor<16x16xi64>) dimensions = [0] 
        %53 = arith.addi %broadcasted_10, %broadcasted_12 : tensor<16x16xi64>
        %54 = tensor.empty() : tensor<16x16xi1>
        %broadcasted_13 = linalg.broadcast ins(%29 : tensor<16xi1>) outs(%54 : tensor<16x16xi1>) dimensions = [1] 
        %55 = arith.extsi %arg16 : i32 to i64
        %56 = arith.muli %31, %55 : i64
        %57 = arith.extsi %arg17 : i32 to i64
        %58:3 = scf.for %arg27 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg28 = %4, %arg29 = %43, %arg30 = %53) -> (tensor<16x16xf32>, tensor<16x16xi64>, tensor<16x16xi64>)  : i32 {
          %61 = arith.muli %arg27, %c16_i32 : i32
          %62 = arith.subi %c64_i32, %61 : i32
          %63 = tensor.empty() : tensor<1x16xi32>
          %64 = linalg.fill ins(%62 : i32) outs(%63 : tensor<1x16xi32>) -> tensor<1x16xi32>
          %65 = arith.cmpi slt, %expanded_4, %64 : tensor<1x16xi32>
          %collapsed_14 = tensor.collapse_shape %65 [[0, 1]] : tensor<1x16xi1> into tensor<16xi1>
          %broadcasted_15 = linalg.broadcast ins(%collapsed_14 : tensor<16xi1>) outs(%54 : tensor<16x16xi1>) dimensions = [0] 
          %66 = arith.andi %broadcasted_13, %broadcasted_15 : tensor<16x16xi1>
          %67 = scf.for %arg31 = %c0 to %c16 step %c1 iter_args(%arg32 = %2) -> (tensor<16x16xf32>) {
            %78 = scf.for %arg33 = %c0 to %c16 step %c1 iter_args(%arg34 = %arg32) -> (tensor<16x16xf32>) {
              %extracted = tensor.extract %arg29[%arg31, %arg33] {DiscreteMemAccess} : tensor<16x16xi64>
              %79 = arith.index_cast %extracted : i64 to index
              %reinterpret_cast_16 = memref.reinterpret_cast %arg2 to offset: [%79], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
              %80 = memref.load %reinterpret_cast_16[%c0] : memref<1xf32, strided<[1], offset: ?>>
              %81 = tensor.empty() : tensor<1x1xf32>
              %82 = linalg.fill ins(%80 : f32) outs(%81 : tensor<1x1xf32>) -> tensor<1x1xf32>
              %inserted_slice = tensor.insert_slice %82 into %arg34[%arg31, %arg33] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
              scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
            } {ExtractedLoadOrStore}
            scf.yield %78 : tensor<16x16xf32>
          } {ExtractedLoadOrStore}
          %68 = arith.select %66, %67, %4 : tensor<16x16xi1>, tensor<16x16xf32>
          %69 = tensor.empty() : tensor<16x16xi8>
          %70 = scf.for %arg31 = %c0 to %c16 step %c1 iter_args(%arg32 = %69) -> (tensor<16x16xi8>) {
            %78 = scf.for %arg33 = %c0 to %c16 step %c1 iter_args(%arg34 = %arg32) -> (tensor<16x16xi8>) {
              %extracted = tensor.extract %arg30[%arg31, %arg33] {DiscreteMemAccess} : tensor<16x16xi64>
              %79 = arith.index_cast %extracted : i64 to index
              %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%79], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
              %80 = memref.load %reinterpret_cast_16[%c0] : memref<1xi8, strided<[1], offset: ?>>
              %81 = tensor.empty() : tensor<1x1xi8>
              %82 = linalg.fill ins(%80 : i8) outs(%81 : tensor<1x1xi8>) -> tensor<1x1xi8>
              %inserted_slice = tensor.insert_slice %82 into %arg34[%arg31, %arg33] [1, 1] [16, 1] : tensor<1x1xi8> into tensor<16x16xi8>
              scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xi8>
            } {ExtractedLoadOrStore}
            scf.yield %78 : tensor<16x16xi8>
          } {ExtractedLoadOrStore}
          %71 = scf.for %arg31 = %c0 to %c16 step %c1 iter_args(%arg32 = %2) -> (tensor<16x16xf32>) {
            %78 = scf.for %arg33 = %c0 to %c16 step %c1 iter_args(%arg34 = %arg32) -> (tensor<16x16xf32>) {
              %79 = arith.index_cast %arg33 : index to i32
              %80 = arith.extsi %79 : i32 to i64
              %81 = arith.addi %34, %80 : i64
              %82 = arith.remsi %81, %c32_i64 : i64
              %83 = arith.muli %82, %57 : i64
              %84 = arith.addi %56, %83 : i64
              %85 = arith.index_cast %arg31 : index to i32
              %86 = arith.addi %85, %61 : i32
              %87 = arith.divsi %86, %c16_i32 : i32
              %88 = arith.extsi %87 : i32 to i64
              %89 = arith.addi %84, %88 : i64
              %90 = arith.index_cast %89 : i64 to index
              %reinterpret_cast_16 = memref.reinterpret_cast %arg5 to offset: [%90], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
              %91 = memref.load %reinterpret_cast_16[%c0] : memref<1xf32, strided<[1], offset: ?>>
              %92 = tensor.empty() : tensor<1x1xf32>
              %93 = linalg.fill ins(%91 : f32) outs(%92 : tensor<1x1xf32>) -> tensor<1x1xf32>
              %inserted_slice = tensor.insert_slice %93 into %arg34[%arg31, %arg33] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
              scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
            } {ExtractedLoadOrStore}
            scf.yield %78 : tensor<16x16xf32>
          } {ExtractedLoadOrStore}
          %72 = arith.sitofp %70 : tensor<16x16xi8> to tensor<16x16xf32>
          %73 = arith.subf %72, %3 : tensor<16x16xf32>
          %74 = arith.mulf %73, %71 : tensor<16x16xf32>
          %75 = linalg.matmul {input_precison = "ieee"} ins(%68, %74 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%arg28 : tensor<16x16xf32>) -> tensor<16x16xf32>
          %76 = arith.addi %arg29, %1 : tensor<16x16xi64>
          %77 = arith.addi %arg30, %1 : tensor<16x16xi64>
          scf.yield %75, %76, %77 : tensor<16x16xf32>, tensor<16x16xi64>, tensor<16x16xi64>
        }
        %59 = arith.extsi %arg15 : i32 to i64
        %60 = scf.for %arg27 = %c0 to %c16 step %c1 iter_args(%arg28 = %2) -> (tensor<16x16xf32>) {
          %61 = scf.for %arg29 = %c0 to %c16 step %c1 iter_args(%arg30 = %arg28) -> (tensor<16x16xf32>) {
            %extracted = tensor.extract %25[%arg27] {DiscreteMemAccess} : tensor<16xi64>
            %62 = arith.muli %59, %extracted : i64
            %63 = arith.index_cast %arg29 : index to i32
            %64 = arith.addi %33, %63 : i32
            %65 = arith.extsi %64 : i32 to i64
            %66 = arith.addi %62, %65 : i64
            %67 = arith.index_cast %66 : i64 to index
            %reinterpret_cast_14 = memref.reinterpret_cast %arg4 to offset: [%67], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            %68 = memref.load %reinterpret_cast_14[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %69 = tensor.empty() : tensor<1x1xf32>
            %70 = linalg.fill ins(%68 : f32) outs(%69 : tensor<1x1xf32>) -> tensor<1x1xf32>
            %inserted_slice = tensor.insert_slice %70 into %arg30[%arg27, %arg29] [1, 1] [16, 1] : tensor<1x1xf32> into tensor<16x16xf32>
            scf.yield {DiscreteMemAccess} %inserted_slice : tensor<16x16xf32>
          } {ExtractedLoadOrStore}
          scf.yield %61 : tensor<16x16xf32>
        } {ExtractedLoadOrStore}
        scf.for %arg27 = %c0 to %c16 step %c1 {
          scf.for %arg28 = %c0 to %c16 step %c1 {
            %extracted = tensor.extract %26[%arg27] {DiscreteMemAccess} : tensor<16xi64>
            %61 = arith.muli %59, %extracted : i64
            %62 = arith.index_cast %arg28 : index to i32
            %63 = arith.addi %33, %62 : i32
            %64 = arith.extsi %63 : i32 to i64
            %65 = arith.addi %61, %64 : i64
            %66 = arith.index_cast %65 : i64 to index
            %67 = arith.cmpi slt, %extracted, %27 : i64
            %68 = arith.cmpi slt, %63, %c32_i32 : i32
            %69 = arith.andi %67, %68 : i1
            %extracted_14 = tensor.extract %58#0[%arg27, %arg28] {DiscreteMemAccess} : tensor<16x16xf32>
            %extracted_15 = tensor.extract %60[%arg27, %arg28] {DiscreteMemAccess} : tensor<16x16xf32>
            %70 = arith.select %69, %extracted_14, %extracted_15 : f32
            %71 = tensor.empty() : tensor<1xf32>
            %72 = linalg.fill ins(%70 : f32) outs(%71 : tensor<1xf32>) -> tensor<1xf32>
            %reinterpret_cast_16 = memref.reinterpret_cast %arg4 to offset: [%66], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
            bufferization.materialize_in_destination %72 in writable %reinterpret_cast_16 : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
          } {ExtractedLoadOrStore}
        } {ExtractedLoadOrStore}
      }
    }
    return
  }
}

