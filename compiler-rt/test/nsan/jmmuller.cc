// RUN: %clangxx_nsan -O0 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t

// RUN: %clangxx_nsan -O1 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t

// RUN: %clangxx_nsan -O2 -g %s -o %t && NSAN_OPTIONS=halt_on_error=1 not %run %t

// This tests J-M Müller's Kahan Challenge:
// http://arith22.gforge.inria.fr/slides/06-gustafson.pdf
//
// The problem is to evaluate `H` at 15, 16, 17, and 9999. The correct
// answer is (1,1,1,1).
// Note that in this case, even though the shadow computation in quad mode is
// also wrong, the inconsistency check shows that there is an issue.

#include <cmath>
#include <cstdio>

double E(double z) {
  return z == 0.0 ? 1.0 : (exp(z) - 1.0) / z;
}

double Q(double x) {
  return fabs(x - sqrt(x * x + 1)) - 1 / (x + sqrt(x * x + 1));
}

__attribute__((noinline)) // Do not constant-fold.
double H(double x) { return E(Q(x * x)); }

int main() {
  constexpr const double kX[] = {15.0, 16.0, 17.0, 9999.0};
  printf("(H(%f), H(%f), H(%f), H(%f)) = (%.8f, %.8f, %.8f, %.8f)\n",
         kX[0], kX[1], kX[2], kX[3],
         H(kX[0]), H(kX[1]), H(kX[2]), H(kX[3]));
  return 0;
}
