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

#include <cstdint>
#include <utility>

bool cmp_greater_equal_i16_i16(int16_t lhs, int16_t rhs) {
  // CHECK:      define dso_local noundef zeroext
  // CHECK-SAME: cmp_greater_equal_i16_i16
  // CHECK-NEXT: %3 = icmp sge i16 %0, %1
  // CHECK-NEXT: ret i1 %3
  // CHECK-NEXT: }
  return std::cmp_greater_equal(lhs, rhs);
}

bool cmp_greater_equal_i32_i32(int32_t lhs, int32_t rhs) {
  // CHECK:      define dso_local noundef zeroext
  // CHECK-SAME: cmp_greater_equal_i32_i32
  // CHECK-NEXT: %3 = icmp sge i32 %0, %1
  // CHECK-NEXT: ret i1 %3
  // CHECK-NEXT: }
  return std::cmp_greater_equal(lhs, rhs);
}

bool cmp_greater_equal_u32_i32(uint32_t lhs, int32_t rhs) {
  // CHECK:      define dso_local noundef zeroext
  // CHECK-SAME: cmp_greater_equal_u32_i32
  // CHECK-NEXT: %3 = icmp slt i32 %1, 0
  // CHECK-NEXT: %4 = icmp uge i32 %0, %1
  // CHECK-NEXT: %5 = or i1 %3, %4
  // CHECK-NEXT: ret i1 %5
  // CHECK-NEXT: }
  return std::cmp_greater_equal(lhs, rhs);
}

bool cmp_greater_equal_i32_u64(int32_t lhs, uint64_t rhs) {
  // CHECK:      define dso_local noundef zeroext
  // CHECK-SAME: cmp_greater_equal_i32_u64
  // CHECK-NEXT: %3 = icmp sgt i32 %0, -1
  // CHECK-NEXT: %4 = zext i32 %0 to i64
  // CHECK-NEXT: %5 = icmp uge i64 %4, %1
  // CHECK-NEXT: %6 = and i1 %3, %5
  // CHECK-NEXT: ret i1 %6
  // CHECK-NEXT: }
  return std::cmp_greater_equal(lhs, rhs);
}

bool cmp_greater_equal_u32_i64(uint32_t lhs, int64_t rhs) {
  // CHECK:      define dso_local noundef zeroext
  // CHECK-SAME: cmp_greater_equal_u32_i64
  // CHECK-NEXT: %3 = icmp slt i64 %1, 0
  // CHECK-NEXT: %4 = zext i32 %0 to i64
  // CHECK-NEXT: %5 = icmp uge i64 %4, %1
  // CHECK-NEXT: %6 = or i1 %3, %5
  // CHECK-NEXT: ret i1 %6
  // CHECK-NEXT: }
  return std::cmp_greater_equal(lhs, rhs);
}

bool cmp_greater_equal_u32_u64(uint32_t lhs, uint64_t rhs) {
  // CHECK:      define dso_local noundef zeroext
  // CHECK-SAME: cmp_greater_equal_u32_u64
  // CHECK-NEXT: %3 = zext i32 %0 to i64
  // CHECK-NEXT: %4 = icmp uge i64 %3, %1
  // CHECK-NEXT: ret i1 %4
  // CHECK-NEXT: }
  return std::cmp_greater_equal(lhs, rhs);
}
