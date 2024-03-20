// RUN: %clangxx_nsan -O2 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <cstddef>

#include "helpers.h"

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes,
                                       size_t bytes_per_line, size_t reserved);

int main() {
  float array[2];
  DoNotOptimize(array);
  array[0] = 1.0;
  array[1] = 2.0;
  __nsan_dump_shadow_mem((const char *)array, sizeof(array), 16, 0);
  // CHECK: {{.*}} f0 f1 f2 f3 f0 f1 f2 f3   (1.00000000000000000000)
  // (2.00000000000000000000)
  return 0;
}
