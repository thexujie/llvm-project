// RUN: %clangxx_nsan -O0 -mllvm -nsan-shadow-type-mapping=dqq -g %s -o %t && NSAN_OPTIONS=halt_on_error=1,enable_loadtracking_stats=1,print_stats_on_exit=1 %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include "helpers.h"

#include <cstdio>
#include <cstring>
#include <memory>

// This tests tracking of loads where the application value has been tampered
// with through type punning.
// nsan resumes by re-extending the original value, and logs the failed tracking
// to stats.

int main() {
  auto d = std::make_unique<double>(2.0);
  printf("%.16f\n", *d);
  DoNotOptimize(d);
  reinterpret_cast<char *>(d.get())[7] = 0;
  DoNotOptimize(d);
  printf("%.16f\n", *d);
  // CHECK: invalid/unknown type for 1/0 loads
  // CHECK: There were 1/0 floating-point loads where the shadow type was invalid/unknown
  // or unknown.
  return 0;
}
