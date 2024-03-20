// RUN: %clangxx_nsan  -O0 -g -DFN=Cube %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan  -O1 -g -DFN=Cube %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan  -O2 -g -DFN=Cube %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O0 -g -DFN=Square %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=Square %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O0 -g -DFN=Inverse %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=Inverse %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out


// Computes the derivative of x -> fn(x) using a finite difference
// approximation:
//     f'(a) = (f(a + da) - f(a)) / da
// https://en.wikipedia.org/wiki/Numerical_differentiation#Finite_differences
// Numerical differentiation is a is a well known case of numerical instability.
// It typically leads to cancellation errors and division issues as `da`
// approaches zero.
// This is similar to `cancellation_libm.cc`, but this variant uses a function
// pointer to a user-defined function instead of a libm function.

#include <cstdio>
#include <cmath>
#define xstr(s) str(s)
#define str(s) #s

static float Square(float x) {
  return x * x;
}

static float Cube(float x) {
  return x * x * x;
}

static float Inverse(float x) {
  return 1.0f / x;
}

__attribute__((noinline))  // To check call stack reporting.
float ComputeDerivative(float(*fn)(float), float a, float da) {
  return (fn(a + da) - fn(a)) / da;
  // CHECK: WARNING: NumericalStabilitySanitizer: inconsistent shadow results while checking return
  // CHECK: float {{ *}}precision (native):
  // CHECK: double{{ *}}precision (shadow):
  // CHECK: {{#0 .*in ComputeDerivative}}
}

int main() {
  for (int i = 7; i < 31; ++i) {
    float step = 1.0f / (1ull << i);
    printf("%s derivative: %.8f\n", xstr(FN), ComputeDerivative(&FN, 0.1f, step));
  }
  return 0;
}

