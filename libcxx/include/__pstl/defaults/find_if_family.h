//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_DEFAULTS_FIND_IF_FAMILY_H
#define _LIBCPP___ALGORITHM_PSTL_DEFAULTS_FIND_IF_FAMILY_H

#include <__config>
#include <__functional/not_fn.h>
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

// This header defines the default implementation for the find_if family of algorithms:
// - find
// - find_if_not
// - any_of
// - all_of
// - none_of
// - is_partitioned

template <class _Backend, class _RawExecutionPolicy>
struct __find_if {
  template <class _Policy, class _ForwardIterator, class _Pred>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator> operator()(
      _Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept = delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __find {
  template <class _Policy, class _ForwardIterator, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) const noexcept {
    using _FindIf = __pstl::__find_if<_Backend, _RawExecutionPolicy>;
    return _FindIf()(
        __policy, std::move(__first), std::move(__last), [&](__iter_reference<_ForwardIterator> __element) {
          return __element == __value;
        });
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __find_if_not {
  template <class _Policy, class _ForwardIterator, class _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept {
    using _FindIf = __pstl::__find_if<_Backend, _RawExecutionPolicy>;
    return _FindIf()(__policy, __first, __last, std::not_fn(__pred));
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __any_of {
  template <class _Policy, class _ForwardIterator, class _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept {
    using _FindIf = __pstl::__find_if<_Backend, _RawExecutionPolicy>;
    auto __res    = _FindIf()(__policy, __first, __last, __pred);
    if (!__res)
      return nullopt;
    return *__res != __last;
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __all_of {
  template <class _Policy, class _ForwardIterator, class _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept {
    using _AnyOf = __pstl::__any_of<_Backend, _RawExecutionPolicy>;
    auto __res   = _AnyOf()(__policy, __first, __last, [&](__iter_reference<_ForwardIterator> __value) {
      return !__pred(__value);
    });
    if (!__res)
      return nullopt;
    return !*__res;
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __none_of {
  template <class _Policy, class _ForwardIterator, class _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept {
    using _AnyOf = __pstl::__any_of<_Backend, _RawExecutionPolicy>;
    auto __res   = _AnyOf()(__policy, __first, __last, __pred);
    if (!__res)
      return nullopt;
    return !*__res;
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __is_partitioned {
  template <class _Policy, class _ForwardIterator, class _Pred>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept {
    using _FindIfNot   = __pstl::__find_if_not<_Backend, _RawExecutionPolicy>;
    auto __maybe_first = _FindIfNot()(__policy, std::move(__first), std::move(__last), __pred);
    if (__maybe_first == nullopt)
      return nullopt;

    __first = *__maybe_first;
    if (__first == __last)
      return true;
    ++__first;
    using _NoneOf = __pstl::__none_of<_Backend, _RawExecutionPolicy>;
    return _NoneOf()(__policy, std::move(__first), std::move(__last), __pred);
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_DEFAULTS_FIND_IF_FAMILY_H
