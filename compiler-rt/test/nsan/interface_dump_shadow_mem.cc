// RUN: %clangxx_nsan -O2 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -fno-builtin -O2 -g  -mllvm -nsan-shadow-type-mapping=dqq %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_nsan -fno-builtin -O2 -g  -mllvm -nsan-shadow-type-mapping=dlq %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This test checks that the sanitizer interface function
// `__nsan_dump_shadow_mem` works correctly.

#include <cstring>
#include <cstdint>
#include <cstdio>


extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes, size_t bytes_per_line, size_t reserved);

int main() {
  char buffer[64];
  int pos = 0;
  // One aligned float.
  const float f = 42.0;
  memcpy(&(buffer[pos]), &f, sizeof(f));
  pos += sizeof(f);
  // One 4-byte aligned double.
  const double d = 35.0;
  memcpy(&(buffer[pos]), &d, sizeof(d));
  pos += sizeof(d);
  // Three uninitialized bytes.
  pos += 3;
  // One char byte.
  buffer[pos] = 'a';
  pos += 1;
  // One long double.
  const long double l = 0.0000000001;
  memcpy(&(buffer[pos]), &l, sizeof(l));
  pos += sizeof(l);
  // One more double, but erase bytes in the middle.
  const double d2 = 53.0;
  memcpy(&(buffer[pos]), &d2, sizeof(d2));
  pos += sizeof(d2);
  uint32_t i = 5;
  memcpy(&(buffer[pos - 5]), &i, sizeof(i));
  // And finally two consecutive floats.
  const float f2 = 43.0;
  memcpy(&(buffer[pos]), &f2, sizeof(f2));
  pos += sizeof(f2);
  const float f3 = 44.0;
  memcpy(&(buffer[pos]), &f3, sizeof(f3));

  __nsan_dump_shadow_mem(buffer, sizeof(buffer), 8, 0);
// CHECK: 0x{{[a-f0-9]*}}:    f0 f1 f2 f3 d0 d1 d2 d3   (42.00000000000000000000)
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    d4 d5 d6 d7 __ __ __ __   (35.00000000000000000000)
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    l0 l1 l2 l3 l4 l5 l6 l7
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    l8 l9 la lb lc ld le lf   (0.00000000010000000000)
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    d0 d1 d2 f0 f1 f2 f3 d7
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    f0 f1 f2 f3 f0 f1 f2 f3   (43.00000000000000000000)  (44.00000000000000000000)
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    __ __ __ __ __ __ __ __
// CHECK-NEXT: 0x{{[a-f0-9]*}}:    __ __ __ __ __ __ __ __
  return 0;
}
