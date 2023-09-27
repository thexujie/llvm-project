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

#include <__algorithm/pstl_backends/gpu_backends/backend.h>

#if defined(_LIBCPP_PSTL_GPU_OFFLOAD)
#  include <__algorithm/pstl_backends/gpu_backends/any_of.h>
#  include <__algorithm/pstl_backends/gpu_backends/fill.h>
#  include <__algorithm/pstl_backends/gpu_backends/find_if.h>
#  include <__algorithm/pstl_backends/gpu_backends/for_each.h>
#  include <__algorithm/pstl_backends/gpu_backends/merge.h>
#  include <__algorithm/pstl_backends/gpu_backends/stable_sort.h>
#  include <__algorithm/pstl_backends/gpu_backends/transform.h>
#  include <__algorithm/pstl_backends/gpu_backends/transform_reduce.h>
#endif

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_H
