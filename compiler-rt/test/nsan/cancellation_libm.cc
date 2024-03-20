// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O1 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out


// NOTE: -fno-math-errno allows clang to emit an intrinsic.

// RUN: %clangxx_nsan -O0 -g %s -o %t -fno-math-errno && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O1 -g %s -o %t -fno-math-errno && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g0 %s -o %t -fno-math-errno && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// Computes the derivative of x -> expf(x) using a finite difference
// aproximation:
//     f'(a) = (f(a + da) - f(a)) / da
// https://en.wikipedia.org/wiki/Numerical_differentiation#Finite_differences
// Numerical differentiation is a is a well known case of numerical instability.
// It typically leads to cancellation errors and division issues as `da`
// approaches zero.

#include <cstdio>
#include <cmath>

// Note that expf is not instrumented, so we cannot detect the numerical
// discrepancy if we do not recognize intrinsics.
__attribute__((noinline))  // To check call stack reporting.
float ComputeDerivative(float a, float da) {
  return (expf(a + da) - expf(a)) / da;
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results while checking return
  // CHECK: float {{ *}}precision (native):
  // CHECK: double{{ *}}precision (shadow):
  // CHECK: {{#0 .*in ComputeDerivative}}
}

int main() {
  for (int i = 1; i < 31; ++i) {
    const float step = 1.0f / (1ull << i);
    printf("derivative (step %f):\n", step);
    printf("    %.8f\n", ComputeDerivative(0.1f, step));
  }
  return 0;
}
