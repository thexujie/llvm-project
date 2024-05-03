// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: foo
__kernel void foo(__global int *i) {
  i[get_group_id(0)] = 1;
}
