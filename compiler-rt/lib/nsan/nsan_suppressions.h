//===-- nsan_suppressions.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines nsan suppression rules.
//===----------------------------------------------------------------------===//

#ifndef NSAN_SUPPRESSIONS_H
#define NSAN_SUPPRESSIONS_H

#include "sanitizer_common/sanitizer_suppressions.h"

namespace __nsan {

extern const char *const kSuppressionNone;
extern const char *const kSuppressionFcmp;
extern const char *const kSuppressionConsistency;

void InitializeSuppressions();

__sanitizer::Suppression *
GetSuppressionForStack(const __sanitizer::StackTrace *Stack,
                       const char *SupprType);

} // namespace __nsan

#endif
