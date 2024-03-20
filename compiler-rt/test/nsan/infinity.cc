// This test case verifies that we handle infinity correctly.

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t
// >%t.out 2>&1

#include "helpers.h"
#include <cstdio>
#include <limits>

__attribute__((noinline))  // To check call stack reporting.
void StoreInf(double* a) {
  DoNotOptimize(a);
  double inf = std::numeric_limits<double>::infinity();
  DoNotOptimize(inf);
  *a = inf;
}

int main() {
  double d;
  StoreInf(&d);
  DoNotOptimize(d);
  printf("%.16f\n", d);
  return 0;
}
