//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_ANY_OF_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_ANY_OF_H

#include <__algorithm/any_of.h>
#include <__algorithm/find_if.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/gpu_backends/backend.h>
#include <__atomic/atomic.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/pair.h>
#include <__utility/terminate_on_exception.h>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
_LIBCPP_HIDE_FROM_ABI bool
__pstl_any_of(__gpu_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  // TODO: Implement GPU backend
  return std::__pstl_any_of<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __pred);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKEND_ANY_OF_H
