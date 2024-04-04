// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: foo
__kernel void foo(__global float4 *f) {
  *f = cross(f[0], f[1]);
}
