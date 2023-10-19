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
#include <__iterator/wrap_iter.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/remove_pointer.h>
#include <__utility/empty.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__pstl_fill(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  // It is only safe to execute fill on the GPU, it the execution policy is
  // parallel unsequenced, as it is the only execution policy allowing
  // SIMD instructions
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value) {
    std::__rewrap_iter(__first, __par_backend::__omp_for_each(std::__unwrap_iter(__first), __last - __first, [&](std::remove_pointer_t<decltype(std::__unwrap_iter(__first))>& e) {
          e = __value;
        }));
    return __empty{};
  }
  // Otherwise, we execute fill on the CPU instead
  else {
    return std::__pstl_fill<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __value);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H
