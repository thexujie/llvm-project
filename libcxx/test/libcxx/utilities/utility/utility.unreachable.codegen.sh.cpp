//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is checking the LLVM IR
// REQUIRES: clang

// RUN: %{cxx} %s %{compile_flags} -O3 -c -S -emit-llvm -o - | %{check-output}

#include <utility>

[[noreturn]] void test() {
  // CHECK:      define dso_local void
  // CHECK-SAME: test
  // CHECK-NEXT: unreachable
  // CHECK-NEXT: }
  std::unreachable();
}
