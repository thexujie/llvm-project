// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 %run %t

// http://cadna.lip6.fr/Examples_Dir/ex2.php
// This is an example where nsan fail to detect an issue. Doing the computations
// in quad instead of double precision does not help in detecting that the
// computation of the determinant is unstable: both double and quad precision
// find it to be positive.

#include <cmath>
#include <cstdio>

extern "C" void __nsan_dump_double(double value);

// Adapted from Fortran: http://cadna.lip6.fr/Examples_Dir/source/ex2.f
__attribute__((noinline)) void Solve(double a, double b, double c) {
  if (a == 0) {
    if (b == 0) {
      if (c == 0) {
        printf("Every complex value is solution.\n");
      } else {
        printf("There is no solution.\n");
      }
    } else {
      double x1 = -c / b;
      printf("'The equation is degenerated. There is one real solution: %f\n",
             x1);
    }
  } else {
    b = b / a;
    c = c / a;
    double d = b * b - 4.0 * c;
    __nsan_dump_double(d); // Print the discriminant shadow value.
    if (d == 0.0) {
      double x1 = -b * 0.5;
      printf("Discriminant is zero. The double solution is %f\n", x1);
    } else if (d > 0) {
      double x1 = (-b - sqrt(d)) * 0.5;
      double x2 = (-b + sqrt(d)) * 0.5;
      printf("There are two real solutions. x1 = %f x2 = %f\n", x1, x2);
    } else {
      double x1 = -b * 0.5;
      double x2 = sqrt(-d) * 0.5;
      printf("There are two complex solutions. z1 = %f %f z2 = %f %f\n", x1, x2,
             x1, -x2);
    }
  }
}

int main() {
  Solve(0.3, -2.1, 3.675);
  return 0;
}
