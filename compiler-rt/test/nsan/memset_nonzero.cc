// RUN: %clangxx_nsan -O0 -mllvm -nsan-shadow-type-mapping=dqq -g %s -o %t && NSAN_OPTIONS=halt_on_error=1,enable_loadtracking_stats=1,print_stats_on_exit=1 %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include "helpers.h"

#include <cstdio>
#include <cstring>

// This tests tracking of loads where the application value has been set to
// a non-zero value in a untyped way (e.g. memset).
// nsan resumes by re-extending the original value, and logs the event to stats.
// Also see `memset_zero.cc`.

int main() {
  double* d = new double(2.0);
  printf("%.16f\n", *d);
  DoNotOptimize(d);
  memset(d, 0x55, sizeof(double));
  DoNotOptimize(d);
  printf("%.16f\n", *d);
// CHECK: There were 0/1 floating-point loads where the shadow type was invalid/unknown.
  return 0;
}
