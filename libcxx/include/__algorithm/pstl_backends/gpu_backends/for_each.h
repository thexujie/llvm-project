//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKNEDS_FOR_EACH_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKNEDS_FOR_EACH_H

#include <__algorithm/for_each.h>
#include <__algorithm/pstl_backends/cpu_backends/backend.h>
#include <__algorithm/pstl_backends/gpu_backends/backend.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/terminate_on_exception.h>
#include <stdio.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy, class _ForwardIterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI void
__pstl_for_each(__gpu_backend_tag, _ForwardIterator __first, _ForwardIterator __last, _Functor __func) {
  // It is only safe to execute for_each on the GPU, it the execution policy is
  // parallel unsequenced, as it is the only execution policy prohibiting throwing
  // exceptions and allowing SIMD instructions
  if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
    std::__par_backend::__parallel_for_simd_1(__first, __last - __first, __func);
  }
  // Else if the excution policy is parallel, we execute for_each on the CPU instead
  else if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                     __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
    std::__terminate_on_exception([&] {
      std::__par_backend::__parallel_for(
          __first, __last, [__func](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            std::__pstl_for_each<__remove_parallel_policy_t<_ExecutionPolicy>>(
                __cpu_backend_tag{}, __brick_first, __brick_last, __func);
          });
    });
    // Else we execute for_each in serial
  } else {
    std::for_each(__first, __last, __func);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_GPU_BACKNEDS_FOR_EACH_H
