//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

struct ConstUncallable {
  template <class T>
  bool operator()(T, T) const& = delete;
  template <class T>
  bool operator()(T, T) & {
    return true;
  }
};

struct NonConstUncallable {
  template <class T>
  bool operator()(T, T) const& {
    return true;
  }
  template <class T>
  bool operator()(T, T) & = delete;
};

void test() {
  {
    auto pair = std::minmax(
        {42, 0, -42},
        ConstUncallable()); //expected-error@*:* {{static assertion failed due to requirement '__is_callable<ConstUncallable, int, int>::value': The comparator has to be callable}}
    (void)pair;
  }
}
