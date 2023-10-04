//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_REDUCE_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_REDUCE_H

#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/gpu_backends/backend.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/transform_reduce.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/operation_traits.h>
#include <__utility/move.h>
#include <__utility/terminate_on_exception.h>
#include <new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _T1, class _T2, class _T3>
struct _LIBCPP_HIDE_FROM_ABI __is_supported_reduction : std::false_type {};

#  define __PSTL_IS_SUPPORTED_REDUCTION(funname)                                                                       \
    template <class _Tp>                                                                                               \
    struct _LIBCPP_HIDE_FROM_ABI __is_supported_reduction<std::funname<_Tp>, _Tp, _Tp> : std::true_type {};            \
    template <class _Tp, class _Up>                                                                                    \
    struct _LIBCPP_HIDE_FROM_ABI __is_supported_reduction<std::funname<>, _Tp, _Up> : std::true_type {};

// __is_trivial_plus_operation already exists
__PSTL_IS_SUPPORTED_REDUCTION(plus)
__PSTL_IS_SUPPORTED_REDUCTION(minus)
__PSTL_IS_SUPPORTED_REDUCTION(multiplies)
__PSTL_IS_SUPPORTED_REDUCTION(logical_and)
__PSTL_IS_SUPPORTED_REDUCTION(logical_or)
__PSTL_IS_SUPPORTED_REDUCTION(bit_and)
__PSTL_IS_SUPPORTED_REDUCTION(bit_or)
__PSTL_IS_SUPPORTED_REDUCTION(bit_xor)

//===----------------------------------------------------------------------===//
// Two input iterators
//===----------------------------------------------------------------------===//

template <class _ExecutionPolicy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _Tp,
          class _BinaryOperation1,
          class _BinaryOperation2>
_LIBCPP_HIDE_FROM_ABI _Tp __pstl_transform_reduce(
    __gpu_backend_tag,
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _Tp __init,
    _BinaryOperation1 __reduce,
    _BinaryOperation2 __transform) {
  // The interface for the function switched between C++17 and C++20
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value && is_arithmetic_v<_Tp> &&
#  if _LIBCPP_STD_VER <= 17
                __libcpp_is_contiguous_iterator<_ForwardIterator1>::value &&
                __libcpp_is_contiguous_iterator<_ForwardIterator2>::value &&
#  endif
                (__is_trivial_plus_operation<_BinaryOperation1, _Tp, _Tp>::value ||
                 __is_supported_reduction<_BinaryOperation1, _Tp, _Tp>::value)) {
    return std::__par_backend::__parallel_for_simd_reduction_2(
        __first1, __first2, __last1 - __first1, __init, __reduce, __transform);
  }
  return std::__pstl_transform_reduce<_ExecutionPolicy>(
      __cpu_backend_tag{}, __first1, __last1, __first2, std::move(__init), __reduce, __transform);
}

//===----------------------------------------------------------------------===//
// One input iterator
//===----------------------------------------------------------------------===//

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
_LIBCPP_HIDE_FROM_ABI _Tp __pstl_transform_reduce(
    __gpu_backend_tag,
    _ForwardIterator __first,
    _ForwardIterator __last,
    _Tp __init,
    _BinaryOperation __reduce,
    _UnaryOperation __transform) {
  // The interface for the function switched between C++17 and C++20
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator>::value && is_arithmetic_v<_Tp> &&
#  if _LIBCPP_STD_VER <= 17
                __libcpp_is_contiguous_iterator<_ForwardIterator>::value &&
#  endif
                (__is_trivial_plus_operation<_BinaryOperation, _Tp, _Tp>::value ||
                 __is_supported_reduction<_BinaryOperation, _Tp, _Tp>::value)) {
    return std::__par_backend::__parallel_for_simd_reduction_1(
        __first, __last - __first, __init, __reduce, __transform);
  }
  return std::__pstl_transform_reduce<_ExecutionPolicy>(
      __cpu_backend_tag{}, __first, __last, std::move(__init), __reduce, __transform);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_REDUCE_H
