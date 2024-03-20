// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// http://cadna.lip6.fr/Examples_Dir/ex4.php
// This example was proposed by J.-M. Muller [1]. The 25 first iterations of the
// following recurrent sequence are computed:
//   U(n+1) = 111 - 1130/U(n) + 3000/(U(n)*U(n-1))
// with U(0) = 5.5 and U(1) = 61/11.
// The exact value for the limit is 6.
// [1] J.-M. Muller, "Arithmetique des ordinateurs", Ed. Masson, 1987.
//
// This checks that nsan correctly detects the instability.


#include <cstdio>

// Adapted from Fortran: http://cadna.lip6.fr/Examples_Dir/source/ex4.f
__attribute__((noinline))  // Prevent constant folding.
void
Ex4(double u_n_minus_1, double u_n, const int end_iter) {
  for (int i = 3; i < end_iter; ++i) {
    const double u_n_plus_1 =
        111.0 - 1130.0 / u_n + 3000.0 / (u_n * u_n_minus_1);
    u_n_minus_1 = u_n;
    u_n = u_n_plus_1;
    printf("U(%i) = %f\n", i, u_n);
// CHECK: #0{{.*}}in Ex4{{.*}}cadna_ex4.cc:[[@LINE-1]]
  }
}

int main() {
  constexpr const double kU1 = 5.5;
  constexpr const double kU2 = 61.0 / 11.0;
  constexpr const double kEndIter = 25;
  Ex4(kU1, kU2, kEndIter);
  return 0;
}
