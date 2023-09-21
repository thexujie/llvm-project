//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_BACKEND_H

#include <__config>
#include <cstddef>

#if defined(_LIBCPP_PSTL_GPU_BACKEND_OMP_OFFLOAD)
#  include <__algorithm/pstl_backends/gpu_backends/omp_offload.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

struct __gpu_backend_tag {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_BACKEND_H
