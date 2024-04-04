// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: foo
__kernel void foo(__global float4 *f) {
  *f = cos(*f);
}
