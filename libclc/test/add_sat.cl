// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: foo
__kernel void foo(__global char *a, __global char *b, __global char *c) {
  *a = add_sat(*b, *c);
}
