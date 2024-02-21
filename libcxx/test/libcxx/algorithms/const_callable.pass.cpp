//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>

struct RvalueRefUncallable {
  template <class T>
  bool operator()(T, T) && = delete;
  template <class T>
  bool operator()(T, T) const& {
    return true;
  }
};

int main(int, char**) { assert(std::minmax({42, 0, -42}, RvalueRefUncallable()).first == -42); }
