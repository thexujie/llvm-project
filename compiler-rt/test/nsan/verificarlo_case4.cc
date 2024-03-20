// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// Case Study #4 from the Verificarlo paper: The loop alternates between
// accumulating extremely large and extremely small values, leading to large
// loss of precision.

#include <cstdio>

using FloatT = double;

__attribute__((noinline)) FloatT Case4(FloatT c, int iterations) {
  for (unsigned i = 0; i < iterations; ++i) {
    if (i % 2 == 0)
      c = c + 1.e6;
    else
      c = c - 1.e-6;
  }
  return c;
  // CHECK: #0 {{.*}} in Case4{{.*}}[[@LINE-1]]
}

int main() {
  for (int iterations = 1; iterations <= 100000000; iterations *= 10) {
    printf("%10i iterations: %f\n", iterations, Case4(-5.e13, iterations));
  }
  return 0;
}

