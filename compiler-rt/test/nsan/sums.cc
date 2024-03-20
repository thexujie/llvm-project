// RUN: %clangxx_nsan -O0 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=NaiveSum -DFLT=float %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=NAIVE,NAIVE-FLOAT < %t.out

// RUN: %clangxx_nsan -O1 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=NaiveSum -DFLT=float %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=NAIVE,NAIVE-FLOAT < %t.out

// RUN: %clangxx_nsan -O2 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=NaiveSum -DFLT=float %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=NAIVE,NAIVE-FLOAT < %t.out
// RUN: %clangxx_nsan -O2 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=NaiveSum -DFLT=double %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=49 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=NAIVE,NAIVE-DOUBLE-QUAD < %t.out
// RUN: %clangxx_nsan -O2 -mllvm -nsan-shadow-type-mapping=dlq -g -DSUM=NaiveSum -DFLT=double %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=49 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=NAIVE,NAIVE-DOUBLE-LONG < %t.out

// RUN: %clangxx_nsan -O0 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=KahanSum -DFLT=float %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 %run %t
// RUN: %clangxx_nsan -O3 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=KahanSum -DFLT=float %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=19 %run %t
// RUN: %clangxx_nsan -O3 -mllvm -nsan-shadow-type-mapping=dqq -g -DSUM=KahanSum -DFLT=double %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=49 %run %t
// RUN: %clangxx_nsan -O3 -mllvm -nsan-shadow-type-mapping=dlq -g -DSUM=KahanSum -DFLT=double %s -o %t && NSAN_OPTIONS=halt_on_error=1,log2_max_relative_error=49 %run %t

#include <iostream>
#include <chrono>
#include <random>
#include <vector>

// A naive, unstable summation.
template <typename T>
__attribute__((noinline))  // To check call stack reporting.
T NaiveSum(const std::vector<T>& values) {
  T sum = 0;
  for (T v : values) {
    sum += v;
  }
  return sum;
  // NAIVE: WARNING: NumericalStabilitySanitizer: inconsistent shadow results while checking return
  // NAIVE-FLOAT: float{{ *}}precision (native):
  // NAIVE-FLOAT: double{{ *}}precision (shadow):
  // NAIVE-DOUBLE-QUAD: double {{ *}}precision (native):
  // NAIVE-DOUBLE-QUAD: __float128{{ *}}precision (shadow):
  // NAIVE-DOUBLE-LONG: double{{ *}}precision (native):
  // NAIVE-DOUBLE-LONG: long double{{ *}}precision (shadow):
  // NAIVE: {{#0 .*in .* NaiveSum}}
}

// Kahan's summation is a numerically stable sum.
// https://en.wikipedia.org/wiki/Kahan_summation_algorithm
template <typename T>
__attribute__((noinline))  // For consistency.
T KahanSum(const std::vector<T>& values) {
  T sum = 0;
  T c = 0;
  for (T v : values) {
    T y = v - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

int main() {
  std::vector<FLT> values;
  constexpr const int kNumValues = 1000000;
  values.reserve(kNumValues);
  // Using a seed to avoid flakiness.
  constexpr uint32_t kSeed = 0x123456;
std::mt19937 gen(kSeed);
  std::uniform_real_distribution<FLT> dis(0.0f, 1000.0f);
  for (int i = 0; i < kNumValues; ++i) {
    values.push_back(dis(gen));
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const auto sum = SUM(values);
  const auto t2 = std::chrono::high_resolution_clock::now();
  printf("sum: %.8f\n", sum);
  std::cout << "runtime: "
            << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                       .count() /
                   1000.0
            << "ms\n";
  return 0;
}

