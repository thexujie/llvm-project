// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=0 %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// http://cadna.lip6.fr/Examples_Dir/ex1.php
// This checks that nsan can detect basic cancellations.

#include <cstdio>

// Adapted from Fortran: http://cadna.lip6.fr/Examples_Dir/source/ex1.f
__attribute__((noinline)) void Ex1(double x, double y) {
  printf("P(%f,%f) = %f\n", x, y, 9.0*x*x*x*x - y*y*y*y + 2.0*y*y);
  // CHECK: #0 {{.*}} in Ex1{{.*}}[[@LINE-1]]
}

int main() {
  Ex1(10864.0, 18817.0);
  // CHECK: #1 {{.*}} in main{{.*}}[[@LINE-1]]
  Ex1(1.0 / 3, 2.0 / 3);
  return 0;
}

