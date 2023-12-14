//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

struct RvalueRefUncallable {
  bool operator()(int, int) && = delete;
  bool operator()(int, int) const &{ return true; }
};

int main(int, char**){
     assert(std::minmax({42, 0, -42}, RvalueRefUncallable{}).first == true);
}