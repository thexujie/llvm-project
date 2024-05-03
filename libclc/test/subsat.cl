// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s

// CHECK: test_subsat_char
__kernel void test_subsat_char(__global char *a, char x, char y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_uchar(__global uchar *a, uchar x, uchar y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_long(__global long *a, long x, long y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_ulong(__global ulong *a, ulong x, ulong y) {
  *a = sub_sat(x, y);
  return;
}