// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=0,log2_max_relative_error=0 %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STOP %s < %t.out

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=0,log2_max_relative_error=0 %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STOP %s < %t.out

// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=REL %s < %t.out

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=REL %s < %t.out

// http://cadna.lip6.fr/Examples_Dir/ex7.php
// This program solves a linear system of order 20 by using Jacobi's method.
// The stopping criterion is
//   || X(n+1) - X(n) || <= eps
// where ||X|| is the maximum norm and eps=0.0001.
//
// This tests that nsan catches two types of errors:
//  - The first one is that the stopping criterion is not stable w.r.t. the
//    precision (STOP). To show this we disable relative error
//    checking and only let the fcmp checker detect the unstable branching.
//  - The second one is that the computations are unstable anyway from the first
//    iteration (REL).

#include <cmath>
#include <cstdio>

// Adapted from Fortran: http://cadna.lip6.fr/Examples_Dir/source/ex7.f

float random1() {
  static int nrand = 23;
  nrand = (nrand * 5363 + 143) % 1387;
  return 2.0 * nrand / 1387.0 - 1.0;
}

int main() {
  constexpr const float kEpsilon = 1e-4;
  constexpr const int kNDims = 20;
  constexpr const int kNIters = 1000;

  float a[kNDims][kNDims];
  float b[kNDims];
  float x[kNDims];
  float y[kNDims];
  const float xsol[kNDims] = {
      1.7,    -4746.89, 50.23, -245.32,  4778.29,  -75.73,  3495.43,
      4.35,   452.98,   -2.76, 8239.24,  3.46,     1000.0,  -5.0,
      3642.4, 735.36,   1.7,   -2349.17, -8247.52, 9843.57,
  };

  for (int i = 0; i < kNDims; ++i) {
    for (int j = 0; j < kNDims; ++j) {
      a[i][j] = random1();
    }
    a[i][i] = a[i][i] + 4.9213648f;
  }

  for (int i = 0; i < kNDims; ++i) {
    float aux = 0.0f;
    for (int j = 0; j < kNDims; ++j) {
      aux = aux + a[i][j]*xsol[j];
    }
    b[i] = aux;
    y[i] = 10.0f;
  }

  int iter = 0;
  for (iter = 0; iter < kNIters; ++iter) {
    float anorm = 0.0f;
    for (int j = 0; j < kNDims; ++j) {
      x[j] = y[j];
    }
    for (int j = 0; j < kNDims; ++j) {
      float aux = b[j];
      for (int k = 0; k < kNDims; ++k) {
        if (k != j) {
          aux = aux - a[j][k]*x[k];
        }
      }
// REL: WARNING: NumericalStabilitySanitizer: inconsistent shadow
// Note: We are not checking the line because nsan detects the issue at the
// `y[j]=` store location in dbg mode, and at the `abs()` location in release
// because the store is optimized out.
      y[j] = aux / a[j][j];

// STOP: WARNING: NumericalStabilitySanitizer: floating-point comparison results depend on precision
// STOP: #0{{.*}}in main{{.*}}cadna_ex7.cc:[[@LINE+1]]
      if (fabsf(x[j]-y[j]) > anorm) {
        anorm = fabsf(x[j]-y[j]);
      }
    }
    printf("iter = %i\n", iter);
// STOP: WARNING: NumericalStabilitySanitizer: floating-point comparison results depend on precision
// STOP: #0{{.*}}in main{{.*}}cadna_ex7.cc:[[@LINE+1]]
    if (anorm < kEpsilon) break;
  }

  printf("niter = %i\n", iter);
  for (int i = 0; i < kNDims; ++i) {
    float aux = -b[i];
    for (int j = 0; j < kNDims; ++j) {
      aux = aux + a[i][j]*y[j];
    }
    printf("x_sol(%2i) = %15.7f (true value : %15.7f), residue(%2i) = %15.7f\n",
           i, y[i], xsol[i], i, aux);
  }

  return 0;
}
