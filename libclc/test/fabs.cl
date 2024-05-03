// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: foo
__kernel void foo(__global float *f) {
  *f = fabs(*f);
}
