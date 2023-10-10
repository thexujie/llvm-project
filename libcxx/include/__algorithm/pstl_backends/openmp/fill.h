//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H

#include <__algorithm/fill.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__config>
#include <__type_traits/is_execution_policy.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
_LIBCPP_HIDE_FROM_ABI void
__pstl_fill(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  // It is only safe to execute fill on the GPU, it the execution policy is
  // parallel unsequenced, as it is the only execution policy allowing
  // SIMD instructions
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value) {
    std::__par_backend::__parallel_for_simd_val_1(__first, __last - __first, __value);
  }
  // Otherwise, we execute fill on the CPU instead
  else {
    std::__pstl_fill<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __value);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H
