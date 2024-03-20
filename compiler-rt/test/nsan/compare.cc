// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This test checks that we warn when a floating-point comparison takes
// different values in the application and shadow domain.

#include <cstdio>
#include <cmath>

// 0.6/0.2 is slightly below 3, so the comparison will fail after a certain
// threshold that depends on the precision of the computation.
__attribute__((noinline))  // To check call stack reporting.
bool DoCmp(double a, double b, double c, double threshold) {
  return c - a / b < threshold;
  // CHECK: WARNING: NumericalStabilitySanitizer: floating-point comparison results depend on precision
  // CHECK: double    {{ *}}precision dec (native): {{.*}}<{{.*}}
  // CHECK: __float128{{ *}}precision dec (shadow): {{.*}}<{{.*}}
  // CHECK: {{#0 .*in DoCmp}}
}

int main() {
  double threshold = 1.0;
  for (int i = 0; i < 60; ++i) {
    threshold /= 2;
    printf("value at threshold %.20f: %i\n", threshold, DoCmp(0.6, 0.2, 3.0, threshold));
  }
  return 0;
}
