//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_DEFAULTS_FOR_EACH_FAMILY_H
#define _LIBCPP___ALGORITHM_PSTL_DEFAULTS_FOR_EACH_FAMILY_H

#include <__algorithm/fill_n.h>
#include <__algorithm/for_each_n.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__pstl/backend_fwd.h>
#include <__utility/empty.h>
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

// This header defines the default implementation for the for_each family of algorithms:
// - for_each_n
// - fill
// - fill_n
// - replace
// - replace_if
// - generate
// - generate_n

template <class _Backend, class _RawExecutionPolicy>
struct __for_each {
  template <class _Policy, class _ForwardIterator, class _Function>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Function&& __func) const noexcept =
      delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __for_each_n {
  template <class _Policy, class _ForwardIterator, class _Size, class _Function>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __size, _Function __func) const noexcept {
    if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _ForEach          = __pstl::__for_each<_Backend, _RawExecutionPolicy>;
      _ForwardIterator __last = __first + __size;
      return _ForEach()(__policy, std::move(__first), std::move(__last), std::move(__func));
    } else {
      // Otherwise, use the serial algorithm to avoid doing two passes over the input
      std::for_each_n(std::move(__first), __size, std::move(__func));
      return __empty{};
    }
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __fill {
  template <class _Policy, class _ForwardIterator, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Tp const& __value) const noexcept {
    using _ForEach = __pstl::__for_each<_Backend, _RawExecutionPolicy>;
    using _Ref     = __iter_reference<_ForwardIterator>;
    return _ForEach()(__policy, std::move(__first), std::move(__last), [&](_Ref __element) { __element = __value; });
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __fill_n {
  template <class _Policy, class _ForwardIterator, class _Size, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __n, _Tp const& __value) const noexcept {
    if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _Fill             = __pstl::__fill<_Backend, _RawExecutionPolicy>;
      _ForwardIterator __last = __first + __n;
      return _Fill()(__policy, std::move(__first), std::move(__last), __value);
    } else {
      // Otherwise, use the serial algorithm to avoid doing two passes over the input
      std::fill_n(std::move(__first), __n, __value);
      return optional<__empty>{__empty{}};
    }
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __replace {
  template <class _Policy, class _ForwardIterator, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Tp const& __old, _Tp const& __new)
      const noexcept {
    using _ReplaceIf = __pstl::__replace_if<_Backend, _RawExecutionPolicy>;
    using _Ref       = __iter_reference<_ForwardIterator>;
    return _ReplaceIf()(
        __policy, std::move(__first), std::move(__last), [&](_Ref __element) { return __element == __old; }, __new);
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __replace_if {
  template <class _Policy, class _ForwardIterator, class _Pred, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty> operator()(
      _Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred, _Tp const& __new_value)
      const noexcept {
    using _ForEach = __pstl::__for_each<_Backend, _RawExecutionPolicy>;
    using _Ref     = __iter_reference<_ForwardIterator>;
    return _ForEach()(__policy, std::move(__first), std::move(__last), [&](_Ref __element) {
      if (__pred(__element))
        __element = __new_value;
    });
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __generate {
  template <class _Policy, class _ForwardIterator, class _Generator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Generator&& __gen) const noexcept {
    using _ForEach = __pstl::__for_each<_Backend, _RawExecutionPolicy>;
    using _Ref     = __iter_reference<_ForwardIterator>;
    return _ForEach()(__policy, std::move(__first), std::move(__last), [&](_Ref __element) { __element = __gen(); });
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __generate_n {
  template <class _Policy, class _ForwardIterator, class _Size, class _Generator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __n, _Generator&& __gen) const noexcept {
    using _ForEachN = __pstl::__for_each_n<_Backend, _RawExecutionPolicy>;
    using _Ref      = __iter_reference<_ForwardIterator>;
    return _ForEachN()(__policy, std::move(__first), __n, [&](_Ref __element) { __element = __gen(); });
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_DEFAULTS_FOR_EACH_FAMILY_H
