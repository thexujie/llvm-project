//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test will fail if the number of devices detected by OpenMP is larger
// than zero but transform is not executed on the device.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -O2 -Wno-pass-failed -fopenmp --offload-arch=native

// REQUIRES: openmp_pstl_backend

#include <algorithm>
#include <cassert>
#include <execution>
#include <vector>
#include <omp.h>

int main(void) {
  // We only run the test if a device is detected by OpenMP
  if (omp_get_num_devices() < 1)
    return 0;

  // Initializing test array
  const int __test_size = 10000;
  std::vector<int> __host(__test_size);
  std::vector<int> __device(__test_size);
  // Should execute on host
  std::transform(std::execution::unseq, __host.begin(), __host.end(), __host.begin(), [](int& h) {
    // Returns true if executed on the host
    h = omp_is_initial_device();
    return h;
  });

  // Finding first index where omp_is_initial_device() returned true
  auto __idx = std::find_if(std::execution::par_unseq, __host.begin(), __host.end(), [](int& n) -> bool { return n; });
  assert(__idx == __host.begin() &&
         "omp_is_initial_device() returned false. std::transform was offloaded but shouldn't be.");

  // Should execute on device
  std::transform(
      std::execution::par_unseq,
      __device.begin(),
      __device.end(),
      __host.begin(),
      __device.begin(),
      [](int& d, int& h) {
        // Should return fals
        d = omp_is_initial_device();
        return h == d;
      });

  // Finding first index where omp_is_initial_device() returned true
  __idx = std::find_if(std::execution::par_unseq, __device.begin(), __device.end(), [](int& n) -> bool { return n; });
  assert(__idx == __device.end() &&
         "omp_is_initial_device() returned true in the target region. std::transform was not offloaded.");
  return 0;
}