// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This tests vector(simd) sanitization.

#include <cstdio>
#include <smmintrin.h>

#include "helpers.h"

int main() {
  double in;
  CreateInconsistency(&in);
  __m128d v = _mm_set1_pd(in);
  DoNotOptimize(in);
  double v2[2];
  _mm_storeu_pd(v2, v);
  // CHECK:{{.*}}inconsistent shadow results while checking store to address
  // CHECK: #0{{.*}}in main{{.*}}[[@LINE-2]]
  DoNotOptimize(v2);
  printf("%f\n", v2[0]);
  // CHECK:{{.*}}inconsistent shadow results while checking call argument #1
  // CHECK: #0{{.*}}in main{{.*}}[[@LINE-2]]
  return 0;
}
