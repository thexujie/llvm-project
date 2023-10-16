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
#include <__utility/empty.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __par_backend {
inline namespace __omp_gpu_backend {

template <class _Tp, class _DifferenceType, class _Up>
_LIBCPP_HIDE_FROM_ABI optional<__empty> __omp_parallel_for_simd_val(
    _Tp* __first, _DifferenceType __n, const _Up& __value) noexcept {
  __omp_gpu_backend::__omp_map_alloc(__first, __n);
#  pragma omp target teams distribute parallel for simd firstprivate(__value)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__first+__i) = __value;
  __omp_gpu_backend::__omp_map_from(__first, __n);
  return __empty{};
}

template <class _ForwardIterator, class _DifferenceType, class _Tp>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__parallel_for_simd_val(_ForwardIterator __first, _DifferenceType __n, const _Tp& __value) noexcept {
  return __omp_gpu_backend::__omp_parallel_for_simd_val(__omp_gpu_backend::__omp_extract_base_ptr(__first), __n, __value);
}

}
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__pstl_fill(__omp_backend_tag, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  // It is only safe to execute fill on the GPU, it the execution policy is
  // parallel unsequenced, as it is the only execution policy allowing
  // SIMD instructions
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value) {
    return std::__par_backend::__parallel_for_simd_val(__first, __last - __first, __value);
  }
  // Otherwise, we execute fill on the CPU instead
  else {
    return std::__pstl_fill<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __value);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_FILL_H
