// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O1 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This tests Rump’s Royal Pain:
// http://arith22.gforge.inria.fr/slides/06-gustafson.pdf
//
// The problem is to evaluate `RumpsRoyalPain(77617, 33096)`. The exact value is
// –0.82739605994682136. Note that in this case, even though the shadow
// computation in quad mode is nowhere near the correct value, the inconsistency
// check shows that there is an issue.

#include <cmath>
#include <cstdio>

__attribute__((noinline)) // Do not constant-fold.
double
RumpsRoyalPain(double x, double y) {
  return 333.75 * pow(y, 6) +
         pow(x, 2) *
             (11 * pow(x, 2) * pow(y, 2) - pow(y, 6) - 121 * pow(y, 4) - 2) +
         5.5 * pow(y, 8) + x / (2 * y);
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results
  // while checking return CHECK: {{#0 .*in RumpsRoyalPain}}
}

int main() {
  constexpr const double kX = 77617;
  constexpr const double kY = 33096;
  printf("RumpsRoyalPain(%f, %f)=%.8f)\n", kX, kY, RumpsRoyalPain(kX, kY));
  return 0;
}
