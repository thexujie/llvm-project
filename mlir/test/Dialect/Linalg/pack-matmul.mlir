// RUN: mlir-opt %s -linalg-pack-matmul=block-factors=32,16,64 -canonicalize -split-input-file | FileCheck %s

func.func @block_matmul(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_matmul(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<4x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = tensor.pack %[[A]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<128x128xf32> -> tensor<4x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.*]] = tensor.empty() : tensor<8x2x64x16xf32>
// CHECK: %[[B_PACKED:.+]] = tensor.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [64, 16]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<128x128xf32> -> tensor<8x2x64x16xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = tensor.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<128x128xf32> -> tensor<4x8x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<4x2x32x64xf32>, tensor<8x2x64x16xf32>) outs(%[[C_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_constant(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst_acc = arith.constant dense<0.0> : tensor<128x128xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%cst_acc : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_constant(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[CST_ACC_PACKED:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8x32x16xf32>
// CHECK-DAG: %[[RES_DST:.+]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  ins({{.*}} : tensor<4x2x32x64xf32>, tensor<8x2x64x16xf32>) outs(%[[CST_ACC_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[RES_DST]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_producer(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%cst : f32) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_producer(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL_DST_PACKED:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[ACC_PACKED:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[FILL_DST_PACKED]] : tensor<4x8x32x16xf32>) -> tensor<4x8x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  ins({{.*}} : tensor<4x2x32x64xf32>, tensor<8x2x64x16xf32>) outs(%[[ACC_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_consumer(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>, %D: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = linalg.add ins(%1, %D : tensor<128x128xf32>, tensor<128x128xf32>)
                  outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %2 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_consumer(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>, %[[D:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[RES_DST:.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  outs({{.*}} : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: %[[ADD_RES:.+]] = linalg.add
// CHECK-SAME:  ins(%[[RES_UNPACKED]], %[[D]] : tensor<128x128xf32>, tensor<128x128xf32>) outs(%[[RES_DST]] : tensor<128x128xf32>)
// CHECK: return %[[ADD_RES]] : tensor<128x128xf32>

// -----

func.func @block_batch_matmul(
    %A: tensor<512x64x128xf32>, %B: tensor<512x128x64xf32>, %C: tensor<512x64x64xf32>) -> tensor<512x64x64xf32> {
  %0 = tensor.empty() : tensor<512x64x64xf32>
  %1 = linalg.batch_matmul ins(%A, %B : tensor<512x64x128xf32>, tensor<512x128x64xf32>)
                           outs(%C : tensor<512x64x64xf32>) -> tensor<512x64x64xf32>
  return %1 : tensor<512x64x64xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>

// CHECK-LABEL: func @block_batch_matmul(
// CHECK-SAME:   %[[A:.+]]: tensor<512x64x128xf32>, %[[B:.+]]: tensor<512x128x64xf32>, %[[C:.+]]: tensor<512x64x64xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<512x2x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = tensor.pack %[[A]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<512x64x128xf32> -> tensor<512x2x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<512x4x2x64x16xf32>
// CHECK: %[[B_PACKED:.+]] = tensor.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [0, 2, 1] inner_dims_pos = [1, 2] inner_tiles = [64, 16]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<512x128x64xf32> -> tensor<512x4x2x64x16xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<512x2x4x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = tensor.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<512x64x64xf32> -> tensor<512x2x4x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<512x2x2x32x64xf32>, tensor<512x4x2x64x16xf32>) outs(%[[C_PACKED]] : tensor<512x2x4x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<512x2x4x32x16xf32> -> tensor<512x64x64xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<512x64x64xf32>
