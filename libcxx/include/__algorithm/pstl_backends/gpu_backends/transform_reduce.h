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
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/transform_reduce.h>
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
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
    return std::__par_backend::__parallel_for_simd_reduction_2(
        std::move(__first1),
        std::move(__first2),
        __last1 - __first1,
        std::move(__init),
        std::move(__reduce),
        [=](__iter_reference<_ForwardIterator1> __in_value_1, __iter_reference<_ForwardIterator1> __in_value_2) {
          return __transform(__in_value_1, __in_value_2);
        });
  } else if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                       __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                       __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
    return std::__terminate_on_exception([&] {
      return __par_backend::__parallel_transform_reduce(
          __first1,
          std::move(__last1),
          [__first1, __first2, __transform](_ForwardIterator1 __iter) {
            return __transform(*__iter, *(__first2 + (__iter - __first1)));
          },
          std::move(__init),
          std::move(__reduce),
          [__first1, __first2, __reduce, __transform](
              _ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last, _Tp __brick_init) {
            return std::__pstl_transform_reduce<__remove_parallel_policy_t<_ExecutionPolicy>>(
                __cpu_backend_tag{},
                __brick_first,
                std::move(__brick_last),
                __first2 + (__brick_first - __first1),
                std::move(__brick_init),
                std::move(__reduce),
                std::move(__transform));
          });
    });
  } else {
    return std::transform_reduce(
        std::move(__first1),
        std::move(__last1),
        std::move(__first2),
        std::move(__init),
        std::move(__reduce),
        std::move(__transform));
  }
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
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
    return std::__par_backend::__parallel_for_simd_reduction_1(
        std::move(__first),
        __last - __first,
        std::move(__init),
        std::move(__reduce),
        [=](__iter_reference<_ForwardIterator> __in_value) { return __transform(__in_value); });
  } else if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                       __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
    return std::__terminate_on_exception([&] {
      return __par_backend::__parallel_transform_reduce(
          std::move(__first),
          std::move(__last),
          [__transform](_ForwardIterator __iter) { return __transform(*__iter); },
          std::move(__init),
          __reduce,
          [__transform, __reduce](auto __brick_first, auto __brick_last, _Tp __brick_init) {
            return std::__pstl_transform_reduce<__remove_parallel_policy_t<_ExecutionPolicy>>(
                __cpu_backend_tag{},
                std::move(__brick_first),
                std::move(__brick_last),
                std::move(__brick_init),
                std::move(__reduce),
                std::move(__transform));
          });
    });
  } else {
    return std::transform_reduce(
        std::move(__first), std::move(__last), std::move(__init), std::move(__reduce), std::move(__transform));
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKENDS_TRANSFORM_REDUCE_H
