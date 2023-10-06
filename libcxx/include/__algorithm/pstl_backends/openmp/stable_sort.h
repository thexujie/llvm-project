//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_STABLE_SORT_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_STABLE_SORT_H

#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__algorithm/stable_sort.h>
#include <__config>
#include <__type_traits/is_execution_policy.h>
#include <__utility/terminate_on_exception.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Comp>
_LIBCPP_HIDE_FROM_ABI void
__pstl_stable_sort(__omp_backend_tag, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp) {
  // TODO: Implement GPU backend.
  std::stable_sort(__first, __last, __comp);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_STABLE_SORT_H
