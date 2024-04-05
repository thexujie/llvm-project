//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_DEFAULTS_STABLE_SORT_FAMILY_H
#define _LIBCPP___ALGORITHM_PSTL_DEFAULTS_STABLE_SORT_FAMILY_H

#include <__config>
#include <__pstl/backend_fwd.h>
#include <__utility/empty.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// This header defines the default implementation for the stable_sort family of algorithms:
// - sort
template <class _Backend, class _RawExecutionPolicy>
struct __stable_sort {
  template <class _Policy, class _RandomAccessIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp&& __comp)
      const noexcept = delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __sort {
  template <class _Policy, class _RandomAccessIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<__empty> operator()(
      _Policy&& __policy, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp&& __comp) const noexcept {
    using _StableSort = __pstl::__stable_sort<_Backend, _RawExecutionPolicy>;
    return _StableSort()(__policy, std::move(__first), std::move(__last), std::forward<_Comp>(__comp));
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_DEFAULTS_STABLE_SORT_FAMILY_H
