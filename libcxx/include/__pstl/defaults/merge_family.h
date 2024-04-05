//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_DEFAULTS_MERGE_FAMILY_H
#define _LIBCPP___ALGORITHM_PSTL_DEFAULTS_MERGE_FAMILY_H

#include <__config>
#include <__pstl/backend_fwd.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// This header defines the default implementation for the merge family of algorithms.
// For now, there are no other algorithms in that family.
template <class _Backend, class _RawExecutionPolicy>
struct __merge {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardOutIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2,
      _ForwardOutIterator __out,
      _Comp&& __comp) const noexcept = delete;
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_DEFAULTS_MERGE_FAMILY_H
