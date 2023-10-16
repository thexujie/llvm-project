//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FIND_IF_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FIND_IF_H

#include <__algorithm/find_if.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__config>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __par_backend {
inline namespace __omp_gpu_backend {

template <class _Tp, class _DifferenceType, class _Predicate>
_LIBCPP_HIDE_FROM_ABI _DifferenceType  __omp_parallel_for_min_idx(_Tp* __first, _DifferenceType __n, _Predicate __pred) noexcept {
  __omp_gpu_backend::__omp_map_to(__first, __n);
  _DifferenceType idx = __n;
#  pragma omp target teams distribute parallel for simd reduction(min:idx)
  for (_DifferenceType __i = 0; __i < __n; ++__i){
    if (__pred(*(__first+__i))) {
      idx = (__i < idx) ? __i : idx;
    }
  }
  __omp_gpu_backend::__omp_map_free(__first, __n);
  return idx;
}

// Extracting the underlying pointer

template <class _ForwardIterator, class _DifferenceType, class _Predicate>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator> __parallel_for_min_idx(_ForwardIterator __first, _DifferenceType __n, _Predicate __pred) noexcept {
  return __first + __omp_gpu_backend::__omp_parallel_for_min_idx(__omp_gpu_backend::__omp_extract_base_ptr(__first), __n, __pred);
}

}
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
__pstl_find_if(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value) {
    return __par_backend::__parallel_for_min_idx(__first, __last - __first, __pred);
  } else {
    return std::__pstl_find_if<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __pred);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FIND_IF_H
