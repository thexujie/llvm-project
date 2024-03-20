// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t

// http://cadna.lip6.fr/Examples_Dir/ex6.php
// The following linear system is solved with the Gaussian elimination method
// with partial pivoting.
//
// This test checks that nsan detects the instability.

#include <cmath>
#include <cstdio>

// Adapted from Fortran: http://cadna.lip6.fr/Examples_Dir/source/ex6.f
int main() {
  constexpr const int kDim = 4;
  constexpr const int kDim1 = 5;

  float xsol[kDim] = {1.0, 1.0, 1.e-8, 1.0};
  float a[kDim][kDim1] = {
      {21.0, 130.0, 0.0, 2.1, 153.1},
      {13.0, 80.0, 4.74e+8, 752.0, 849.74},
      {0.0, -0.4, 3.9816e+8, 4.2, 7.7816},
      {0.0, 0.0, 1.7, 9.0e-9, 2.6e-8},
  };

  for (int i = 0; i < kDim - 1; ++i) {
    float pmax = 0.0 ;
    int ll;
    for (int j = i; j < kDim; ++j) {
      const float a_j_i = a[j][i];
      if (fabsf(a_j_i) > pmax) {
        pmax = abs(a_j_i);
        ll = j;
      }
    }

    if (ll != i) {
      for (int j = i; j < kDim1; ++j) {
        std::swap(a[i][j], a[ll][j]);
      }
    }

    const float a_i_i = a[i][i];
    for (int j = i + 1; j < kDim1; ++j) {
      a[i][j] = a[i][j] / a_i_i;
    }

    for (int k = i + 1; k < kDim; ++k) {
      const float a_k_i = a[k][i];
      for (int j = i + 1; j < kDim1; ++j) {
        a[k][j] = a[k][j] - a_k_i * a[i][j];
      }
    }
  }

  a[kDim - 1][kDim1 - 1] = a[kDim - 1][kDim1 - 1] / a[kDim - 1][kDim - 1];
  for (int i = kDim - 2; i >= 0; --i) {
    for (int j = i + 1; j < kDim; ++j) {
      a[i][kDim1 - 1] = a[i][kDim1 - 1] - a[i][j] * a[j][kDim1 - 1];
    }
  }
  for (int i = 0; i < kDim; ++i) {
    printf("x_sol[%i] = %g (true value : %g)\n", i, a[i][kDim1 - 1], xsol[i]);
  }
  return 0;
}
