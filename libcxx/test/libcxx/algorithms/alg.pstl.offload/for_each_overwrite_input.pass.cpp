//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that you can overwrite the input in std::for_each. If the
// result was not copied back from the device to the host, this test would fail.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -fopenmp --offload-arch=native

// REQUIRES: openmp_pstl_backend

#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <vector>

template <class _Tp, class _Predicate, class _Up>
void overwrite(_Tp& __data, _Predicate __pred, const _Up& __value) {
  // This function assumes that __pred will never be the identity transformation
  // Filling array with __value
  std::fill(std::execution::par_unseq, __data.begin(), __data.end(), __value);

  // Updating the array with a lambda
  std::for_each(std::execution::par_unseq, __data.begin(), __data.end(), __pred);

  // Asserting that no elements have the intial value
  auto __idx = std::find_if(
      std::execution::par_unseq, __data.begin(), __data.end(), [&, __value](decltype(__data[0])& n) -> bool {
        return n == __value;
      });
  assert(__idx == __data.end());
}

int main(void) {
  const int __test_size = 10000;
  // Testing with vector of doubles
  {
    std::vector<double> __v(__test_size);
    overwrite(__v, [&](double& __n) { __n *= __n; }, 2.0);
  }
  // Testing with vector of integers
  {
    std::vector<int> __v(__test_size);
    overwrite(__v, [&](int& __n) { __n *= __n; }, 2);
  }
  // Testing with array of doubles
  {
    std::array<double, __test_size> __a;
    overwrite(__a, [&](double& __n) { __n *= __n; }, 2.0);
  }
  // Testing with array of integers
  {
    std::array<int, __test_size> __a;
    overwrite(__a, [&](int& __n) { __n *= __n; }, 2);
  }
  return 0;
}