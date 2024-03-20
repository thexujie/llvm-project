// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t

// http://cadna.lip6.fr/Examples_Dir/ex3.php
// The determinant of Hilbert's matrix (11x11) without pivoting strategy is
// computed. After triangularization, the determinant is the product of the
// diagonal elements.

#include <cstdio>

// Adapted from Fortran: http://cadna.lip6.fr/Examples_Dir/source/ex3.f
int main() {
  constexpr const int kN = 11;
  double amat[kN][kN];
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kN; ++j) {
      // Hilbert's matrix is defined by: a(i,j) = 1/(i+j+1),
      // where i and j are zero-based.
      amat[i][j] = 1.0 / (i + j + 1);
      printf("%.3f, ", amat[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  double det = 1.0;
  for (int i = 0; i < kN - 1; ++i) {
    printf("Pivot number %2i = %f\n", i, amat[i][i]);
    det = det * amat[i][i];
    const double aux = 1.0 / amat[i][i];
    for (int j = i + 1; j < kN; ++j) {
      amat[i][j] = amat[i][j] * aux;
    }

    for (int j = i + 1; j < kN; ++j) {
      const double aux = amat[j][i];
      for (int k = i + 1; k < kN; ++k) {
        amat[j][k] = amat[j][k] - aux * amat[i][k];
      }
    }
  }

  constexpr const int kLastElem = kN-1;
  const double last_pivot = amat[kLastElem][kLastElem];
  printf("Pivot number %2i = %f\n", kLastElem, last_pivot);
  det = det * last_pivot;
  printf("Determinant     = %.12g\n", det);
  return 0;
}
