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

// ADDITIONAL_COMPILE_FLAGS: -O2 -Wno-pass-failed -fopenmp --offload-arch=native

// REQUIRES: openmp_pstl_backend

#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <vector>

template <class _Tp>
void check_find_if(_Tp& __data) {
  const int __len = __data.end() - __data.begin();
  // Setting all elements to two except for the indexes in __idx
  int __idx[11] = {
      0, __len / 10, __len / 9, __len / 8, __len / 7, __len / 6, __len / 5, __len / 4, __len / 3, __len / 2, __len - 1};
  std::fill(std::execution::par_unseq, __data.begin(), __data.end(), 2);
  for (auto __i : __idx) {
    __data[__i]--;
  };
  // Asserting that the minimas are found in the correct order
  for (auto __i : __idx) {
    auto __found_min = std::find_if(
        std::execution::par_unseq, __data.begin(), __data.end(), [&](decltype(__data[0])& n) -> bool { return n < 2; });
    assert(__found_min == (__data.begin() + __i));
    // Incrementing the minimum, so the next one can be found
    (*__found_min)++;
  }
}

int main(void) {
  const int __test_size = 10000;
  // Testing with vector of doubles
  {
    std::vector<double> __v(__test_size);
    check_find_if(__v);
  }
  // Testing with vector of integers
  {
    std::vector<int> __v(__test_size);
    check_find_if(__v);
  }
  // Testing with array of doubles
  {
    std::array<double, __test_size> __a;
    check_find_if(__a);
  }
  // Testing with array of integers
  {
    std::array<int, __test_size> __a;
    check_find_if(__a);
  }
  return 0;
}