//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_H

#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/gpu_backends/backend.h>
#include <__algorithm/transform.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/terminate_on_exception.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator __pstl_transform(
    __gpu_backend_tag,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator __result,
    _UnaryOperation __op) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value) {
    // While the CPU backend captures by reference, [&], that is not valid when
    // offloading to the GPU. Therefore we must capture by value, [=].
    return std::__par_backend::__parallel_for_simd_2(__first, __last - __first, __result, __op);
  }
  return std::__pstl_transform<_ExecutionPolicy>(__cpu_backend_tag{}, __first, __last, __result, __op);
}

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _ForwardOutIterator,
          class _BinaryOperation,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator __pstl_transform(
    __gpu_backend_tag,
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardOutIterator __result,
    _BinaryOperation __op) {
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator1>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator2>::value &&
                __libcpp_is_contiguous_iterator<_ForwardOutIterator>::value) {
    // While the CPU backend captures by reference, [&], that is not valid when
    // offloading to the GPU. Therefore we must capture by value, [=].
    return std::__par_backend::__parallel_for_simd_3(__first1, __last1 - __first1, __first2, __result, __op);
  }
  return std::__pstl_transform<_ExecutionPolicy>(__cpu_backend_tag{}, __first1, __last1, __first2, __result, __op);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_H
