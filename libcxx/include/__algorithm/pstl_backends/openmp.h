//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_H

#include <__config>

#include <__algorithm/pstl_backends/openmp/backend.h>

#if defined(_LIBCPP_PSTL_BACKEND_OPENMP)
#  include <__algorithm/pstl_backends/openmp/any_of.h>
#  include <__algorithm/pstl_backends/openmp/fill.h>
#  include <__algorithm/pstl_backends/openmp/find_if.h>
#  include <__algorithm/pstl_backends/openmp/for_each.h>
#  include <__algorithm/pstl_backends/openmp/merge.h>
#  include <__algorithm/pstl_backends/openmp/stable_sort.h>
#  include <__algorithm/pstl_backends/openmp/transform.h>
#  include <__algorithm/pstl_backends/openmp/transform_reduce.h>
#endif

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_H
