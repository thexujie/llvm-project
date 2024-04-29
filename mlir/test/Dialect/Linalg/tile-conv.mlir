// RUN: mlir-opt %s -transform-interpreter -canonicalize -split-input-file | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 1)>

func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile_using_for %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: func @conv
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[KH:.*]] = memref.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[KW:.*]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[H:.*]] = memref.dim %[[ARG2]], %[[C0]]
//   CHECK-DAG:   %[[W:.*]] = memref.dim %[[ARG2]], %[[C1]]
//       CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[H]] step %[[C2]]
//       CHECK:     scf.for %[[J:.*]] = %[[C0]] to %[[W]] step %[[C3]]
//   CHECK-DAG:     %[[T4:.*]] = affine.min #[[MAP0]](%[[I]])[%[[H]]]
//   CHECK-DAG:       %[[T5:.*]] = affine.min #[[MAP1]](%[[J]])[%[[W]]]
//   CHECK-DAG:       %[[T6:.*]] = affine.apply #[[MAP2]](%[[T4]])[%[[KH]]]
//   CHECK-DAG:       %[[T7:.*]] = affine.apply #[[MAP2]](%[[T5]])[%[[KW]]]
//   CHECK-DAG:       %[[SVIN:.*]] = memref.subview %[[ARG0]][%[[I]], %[[J]]] [%[[T6]], %[[T7]]]
//   CHECK-DAG:       %[[SVKER:.*]] = memref.subview %[[ARG1]][0, 0] [%[[KH]], %[[KW]]]
//   CHECK-DAG:       %[[SVOUT:.*]] = memref.subview %[[ARG2]][%[[I]], %[[J]]] [%[[T4]], %[[T5]]]
//       CHECK:       linalg.conv_2d
//  CHECK-SAME:         ins(%[[SVIN]], %[[SVKER]]
//  CHECK-SAME:         outs(%[[SVOUT]]

// -----

func.func @depthwise_conv_2D(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>, %arg2 : memref<?x?x?x?x?xf32>) {
  linalg.depthwise_conv_nd ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_nd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:3 = transform.structured.tile_using_for %0 [2, 3, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//       CHECK: func @depthwise_conv_2D
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<?x?x?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: memref<?x?x?x?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[BATCH:.*]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[CHANNELS:.*]] = memref.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[MULTIPLIER:.*]] = memref.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[KW:.*]] = memref.dim %[[ARG1]], %[[C2]]
//   CHECK-DAG:   %[[KH:.*]] = memref.dim %[[ARG1]], %[[C3]]
//   CHECK-DAG:   %[[W:.*]] = memref.dim %[[ARG2]], %[[C3]]
//   CHECK-DAG:   %[[H:.*]] = memref.dim %[[ARG2]], %[[C4]]
//       CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[BATCH]] step %[[C2]]
//       CHECK:     scf.for %[[J:.*]] = %[[C0]] to %[[W]] step %[[C3]]
//       CHECK:       scf.for %[[K:.*]] = %[[C0]] to %[[H]] step %[[C3]]
//   CHECK-DAG:       %[[T4:.*]] = affine.min #[[MAP0]](%[[I]])[%[[BATCH]]]
//   CHECK-DAG:       %[[T5:.*]] = affine.min #[[MAP1]](%[[J]])[%[[W]]]
//   CHECK-DAG:       %[[T6:.*]] = affine.min #[[MAP1]](%[[K]])[%[[H]]]
//   CHECK-DAG:       %[[T7:.*]] = affine.apply #[[MAP2]](%[[T5]])[%[[KW]]]
//   CHECK-DAG:       %[[T8:.*]] = affine.apply #[[MAP2]](%[[T6]])[%[[KH]]]
//   CHECK-DAG:       %[[SVIN:.*]] = memref.subview %[[ARG0]][%[[I]], 0, %[[J]], %[[K]]] [%[[T4]], %[[CHANNELS]], %[[T7]], %[[T8]]]
//   CHECK-DAG:       %[[SVKER:.*]] = memref.subview %[[ARG1]][0, 0, 0, 0] [%[[CHANNELS]], %[[MULTIPLIER]], %[[KW]], %[[KH]]]
//   CHECK-DAG:       %[[SVOUT:.*]] = memref.subview %[[ARG2]][%[[I]], 0, 0, %[[J]], %[[K]]] [%[[T4]], %[[CHANNELS]], %[[MULTIPLIER]], %[[T5]], %[[T6]]]
//       CHECK:       linalg.depthwise_conv_nd {channel_first = true}
//  CHECK-SAME:         ins(%[[SVIN]], %[[SVKER]]
//  CHECK-SAME:         outs(%[[SVOUT]]
