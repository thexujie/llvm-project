//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<_RealType = double>
// class uniform_real_distribution;

// result_type must be floating type, int type is unsupported

#include <random>

// expected-error@*:* {{static assertion failed due to requirement '__libcpp_random_is_valid_realtype<int>::value': RealType must be a supported floating-point type}}
struct test_random : public std::uniform_real_distribution<int> {};
