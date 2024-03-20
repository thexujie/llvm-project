// RUN: %clangxx_nsan -O0 -g -DIMPL=Naive -mllvm -nsan-instrument-fcmp=0 %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t
// RUN: %clangxx_nsan -O2 -g -DIMPL=Naive -mllvm -nsan-instrument-fcmp=0 %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t
// RUN: %clangxx_nsan -O0 -g -DIMPL=Better1 -mllvm -nsan-instrument-fcmp=0 %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t
// RUN: %clangxx_nsan -O2 -g -DIMPL=Better1 -mllvm -nsan-instrument-fcmp=0 %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t
// RUN: %clangxx_nsan -O0 -g -DIMPL=Better2 -mllvm -nsan-instrument-fcmp=0 %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t
// RUN: %clangxx_nsan -O2 -g -DIMPL=Better2 -mllvm -nsan-instrument-fcmp=0 %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t

// This tests a few cancellations from the implementations of the function
// presented in: https://people.eecs.berkeley.edu/~wkahan/JAVAhurt.pdf, page 27.
// All three functions have varying degrees of cancellation, none of which
// lead to catastrophic errors.


#include <cmath>
#include <cstdio>

// This never loses more than 1/2 of the digits.
static double Naive(const double X) __attribute__((noinline)) {
  double Y, Z;
  Y = X - 1.0;
  Z = exp(Y);
  if (Z != 1.0)
    Z = Y / (Z - 1.0);
  return Z;
}

static double Better1(const double X) __attribute__((noinline)) {
  long double Y, Z;
  Y = X - 1.0;
  Z = exp(Y);
  if (Z != 1.0)
    Z = Y / (Z - 1.0);
  return Z;
}

// This is precise to a a few ulps.
static double Better2(const double X) __attribute__((noinline)) {
  double Y, Z;
  Y = X - 1.0;
  Z = exp(Y);
  if (Z != 1.0)
    Z = log(Z) / (Z - 1.0);
  return Z;
}

int main() {
  for (int i = 7; i < 31; ++i) {
    const double x = 1.0 + 1.0 / (1ull << i);
    printf("value at %.16f:\n", x);
    printf("    %.16f\n", IMPL(x));
  }
  return 0;
}
