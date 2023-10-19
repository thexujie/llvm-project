//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_TRANSFORM_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_TRANSFORM_H

#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/openmp/backend.h>
#include <__algorithm/transform.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/wrap_iter.h>
#include <__type_traits/is_execution_policy.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

//===----------------------------------------------------------------------===//
// OpenMP implementations of transform for one and two input iterators and one
// output iterator
//===----------------------------------------------------------------------===//

template <class _Tp, class _DifferenceType, class _Up, class _Function>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__omp_transform(_Tp* __first1, _DifferenceType __n, _Up* __first2, _Function __f) noexcept {
  __par_backend::__omp_map_alloc(__first2, __n);
  __par_backend::__omp_map_to(__first1, __n);
#  pragma omp target teams distribute parallel for simd
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__first2 + __i) = __f(*(__first1 + __i));
  __par_backend::__omp_map_from(__first2, __n);
  __par_backend::__omp_map_free(__first1, __n);
  return __empty{};
}

template <class _Tp, class _DifferenceType, class _Up, class _Vp, class _Function>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__omp_transform(_Tp* __first1, _DifferenceType __n, _Up* __first2, _Vp* __first3, _Function __f) noexcept {
  __par_backend::__omp_map_to(__first1, __n);
  __par_backend::__omp_map_to(__first2, __n);
  __par_backend::__omp_map_alloc(__first3, __n);
#  pragma omp target teams distribute parallel for simd
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    *(__first3 + __i) = __f(*(__first1 + __i), *(__first2 + __i));
  __par_backend::__omp_map_free(__first1, __n);
  __par_backend::__omp_map_free(__first2, __n);
  __par_backend::__omp_map_from(__first3, __n);
  return __empty{};
}

template <class _ExecutionPolicy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> __pstl_transform(
    __omp_backend_tag,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator __result,
    _UnaryOperation __op) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
                __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value) {
    std::__omp_transform(std::__unwrap_iter(__first), __last - __first, std::__unwrap_iter(__result), __op);
    return __result + (__last - __first);
  }
  // If it is not safe to offload to the GPU, we rely on the CPU backend.
  return std::__pstl_transform<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __result, __op);
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _BinaryOperation,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> __pstl_transform(
    __omp_backend_tag,
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardOutIterator __result,
    _BinaryOperation __op) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __libcpp_is_contiguous_iterator<_ForwardIterator1>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator2>::value &&
                __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value) {
    std::__omp_transform(std::__unwrap_iter(__first1), __last1 - __first1, std::__unwrap_iter(__first2), std::__unwrap_iter(__result), __op);
    return __result + (__last1 - __first1);
  }
  // If it is not safe to offload to the GPU, we rely on the CPU backend.
  return std::__pstl_transform<_ExecutionPolicy>(__cpu_backend_tag{}, __first1, __last1, __first2, __result, __op);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_TRANSFORM_H
