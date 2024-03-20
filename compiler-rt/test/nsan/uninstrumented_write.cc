// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=0 %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This test load checking. Inconsistencies on load can happen when
// uninstrumented code writes to memory.

#include "helpers.h"

#include <cstdio>
#include <memory>

int main() {
  auto d = std::make_unique<double>(2.0);
  printf("%.16f\n", *d);
  DoNotOptimize(d);
  // Sneakily change the sign bit.
  asm volatile("xorb $0x80, 7(%0)" : : "r"(d.get()));
  printf("%.16f\n", *d);
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results
  // while checking call argument #1 CHECK: {{#0 .*in main}}
  return 0;
}
