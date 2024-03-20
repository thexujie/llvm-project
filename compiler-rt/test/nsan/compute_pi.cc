// RUN: %clangxx_nsan -O0 -mllvm -nsan-shadow-type-mapping=dqq -g -DRECURRENCE=Good %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=10 %run %t

// RUN: %clangxx_nsan -O1 -mllvm -nsan-shadow-type-mapping=dqq -g -DRECURRENCE=Good %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=10 %run %t

// RUN: %clangxx_nsan -O2 -mllvm -nsan-shadow-type-mapping=dqq -g0 -DRECURRENCE=Good %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=10 %run %t

// RUN: %clangxx_nsan -O0 -mllvm -nsan-shadow-type-mapping=dqq -g -DRECURRENCE=Bad %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=10 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O1 -mllvm -nsan-shadow-type-mapping=dqq -g -DRECURRENCE=Bad %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=10 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -mllvm -nsan-shadow-type-mapping=dqq -g0 -DRECURRENCE=Bad %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=10 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This is the Archimedes algorithm for computing pi, starting from a hexagon
// and doubling the number of edges at every iteration.
// https://en.wikipedia.org/wiki/Floating-point_arithmetic#Minimizing_the_effect_of_accuracy_problems

#include <cstdio>
#include <cmath>

__attribute__((noinline))  // To check call stack reporting.
double Bad(double ti) {
  return (sqrt(ti * ti + 1) - 1) / ti;
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results
  // CHECK: double     {{ *}}precision (native):
  // CHECK: __float128 {{ *}}precision (shadow):
  // CHECK: {{#0 .*in Bad}}
}

// This is a better equivalent that does not have the unstable cancellation.
__attribute__((noinline))  // For consistency.
double Good(double ti) {
  return ti / (sqrt(ti * ti + 1) + 1);
}

int main() {
  double ti = 1/sqrt(3);  // t0;
  for (int i = 0; i < 60; ++i) {
    printf("%2i   pi= %.16f\n", i, 6.0 * (1ull << i) * ti);
    ti = RECURRENCE(ti);
  }
  return 0;
}
