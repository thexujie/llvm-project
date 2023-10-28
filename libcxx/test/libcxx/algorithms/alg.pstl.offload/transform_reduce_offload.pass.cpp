//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but transform_reduce is not executed on the device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -O2 -Wno-pass-failed -fopenmp

// REQUIRES: openmp_pstl_backend

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>
#include <functional>

int main(void) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  // Initializing test array
  const int __test_size = 10000;
  std::vector<int> __v(__test_size);
  std::vector<int> __w(__test_size);
  std::for_each(std::execution::par_unseq, __v.begin(), __v.end(), [](int& n) { n = !omp_is_initial_device(); });

  std::for_each(std::execution::par_unseq, __w.begin(), __w.end(), [](int& n) { n = !omp_is_initial_device(); });

  int result = std::transform_reduce(
      std::execution::par_unseq, __v.begin(), __v.end(), __w.begin(), (int)0, std::plus{}, [](int& n, int& m) {
        return n + m + omp_is_initial_device();
      });
  assert(result == 2 * __test_size &&
         "omp_is_initial_device() returned true in the target region. std::transform_reduce was not offloaded.");
  return 0;
}