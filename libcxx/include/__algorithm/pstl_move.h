//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_MOVE_H
#define _LIBCPP___ALGORITHM_PSTL_MOVE_H

#include <__config>
#include <__iterator/cpp17_iterator_concepts.h>
#include <__pstl/configuration.h>
#include <__pstl/run_backend.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _ForwardOutIterator,
          class _RawPolicy                                    = __remove_cvref_t<_ExecutionPolicy>,
          enable_if_t<is_execution_policy_v<_RawPolicy>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardOutIterator
move(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __out) {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardIterator);
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(_ForwardOutIterator);
  _LIBCPP_REQUIRE_CPP17_OUTPUT_ITERATOR(_ForwardOutIterator, decltype(std::move(*__first)));
  using _Implementation = __pstl::__move<__pstl::__configured_backend, _RawPolicy>;
  return __pstl::__run_backend<_Implementation>(
      std::forward<_ExecutionPolicy>(__policy), std::move(__first), std::move(__last), std::move(__out));
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_MOVE_H
