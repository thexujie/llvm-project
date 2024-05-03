// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: foo
__kernel void foo(__global int4 *x, __global float4 *y) {
  *x = convert_int4(*y);
}
