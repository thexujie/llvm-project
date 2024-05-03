// RUN: %clang -emit-llvm -S %s

#if defined(cl_khr_fp64)

__kernel void foo(__global float4 *x, __global double4 *y) {
  x[1] = rsqrt(x[0]);
  y[1] = rsqrt(y[0]);
}

#endif
