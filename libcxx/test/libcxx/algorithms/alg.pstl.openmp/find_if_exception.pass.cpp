//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that we can run code with exceptions on the device, as
// long as no exception is not thrown.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS:  -Wno-openmp-target-exception -fexceptions --offload-arch=native -L%{lib}/../../lib -lomptarget -L%{lib}/../../projects/openmp/libomptarget/ -lomptarget.devicertl

// REQUIRES: openmp_pstl_backend

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

bool is_even(int& i) {
  try {
    if ((i % 2) == 0) {
      return true;
    } else {
      throw false;
    }
  } catch (bool b) {
    return b;
  }
}
#pragma omp declare target indirect to(is_even)

int main(int, char**) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  const int test_size = 10000;
  std::vector<int> v(test_size, 2);

  // Providing for_each a function pointer
  auto idx = std::find_if(std::execution::par_unseq, v.begin(), v.end(), is_even);

  assert(idx == v.begin() && "std::find_if(std::execution::par_unseq,...) does not support exception handling.");
  return 0;
}
