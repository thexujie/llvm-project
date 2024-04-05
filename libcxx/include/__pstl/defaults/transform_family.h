//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_DEFAULTS_TRANSFORM_FAMILY_H
#define _LIBCPP___ALGORITHM_PSTL_DEFAULTS_TRANSFORM_FAMILY_H

#include <__algorithm/copy_n.h>
#include <__config>
#include <__functional/identity.h>
#include <__pstl/backend_fwd.h>
#include <__utility/empty.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// This header defines the default implementation for the transform family of algorithms:
// - replace_copy_if
// - replace_copy
// - move
// - copy
// - copy_n
// - rotate_copy

template <class _Backend, class _RawExecutionPolicy>
struct __transform {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _ForwardOutIterator __out,
             _UnaryOperation&& __op) const noexcept = delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __transform_binary {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _ForwardOutIterator,
            class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardOutIterator __out,
             _BinaryOperation&& __op) const noexcept = delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __replace_copy_if {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _Pred, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _ForwardOutIterator __out,
             _Pred&& __pred,
             _Tp const& __new_value) const noexcept {
    using _Transform = __pstl::__transform<_Backend, _RawExecutionPolicy>;
    using _Ref       = __iter_reference<_ForwardIterator>;
    auto __res = _Transform()(__policy, std::move(__first), std::move(__last), std::move(__out), [&](_Ref __element) {
      return __pred(__element) ? __new_value : __element;
    });
    if (__res == nullopt)
      return nullopt;
    return __empty{};
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __replace_copy {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _ForwardOutIterator __out,
             _Tp const& __old_value,
             _Tp const& __new_value) const noexcept {
    using _ReplaceCopyIf = __pstl::__replace_copy_if<_Backend, _RawExecutionPolicy>;
    using _Ref           = __iter_reference<_ForwardIterator>;
    return _ReplaceCopyIf()(
        __policy,
        std::move(__first),
        std::move(__last),
        std::move(__out),
        [&](_Ref __element) { return __element == __old_value; },
        __new_value);
  }
};

// TODO: Use the std::copy/move shenanigans to forward to std::memmove
//       Investigate whether we want to still forward to std::transform(policy)
//       in that case for the execution::par part, or whether we actually want
//       to run everything serially in that case.
template <class _Backend, class _RawExecutionPolicy>
struct __move {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __out) const noexcept {
    using _Transform = __pstl::__transform<_Backend, _RawExecutionPolicy>;
    return _Transform()(__policy, std::move(__first), std::move(__last), std::move(__out), [&](auto&& __element) {
      return std::move(__element);
    });
  }
};

// TODO: Use the std::copy/move shenanigans to forward to std::memmove
template <class _Backend, class _RawExecutionPolicy>
struct __copy {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __out) const noexcept {
    using _Transform = __pstl::__transform<_Backend, _RawExecutionPolicy>;
    return _Transform()(__policy, std::move(__first), std::move(__last), std::move(__out), __identity());
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __copy_n {
  template <class _Policy, class _ForwardIterator, class _Size, class _ForwardOutIterator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy, _ForwardIterator __first, _Size __n, _ForwardOutIterator __out) const noexcept {
    if constexpr (__has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _Copy             = __pstl::__copy<_Backend, _RawExecutionPolicy>;
      _ForwardIterator __last = __first + __n;
      return _Copy()(__policy, std::move(__first), std::move(__last), std::move(__out));
    } else {
      // Otherwise, use the serial algorithm to avoid doing two passes over the input
      return std::copy_n(std::move(__first), __n, std::move(__out));
    }
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __rotate_copy {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __middle,
             _ForwardIterator __last,
             _ForwardOutIterator __out) const noexcept {
    using _Copy       = __pstl::__copy<_Backend, _RawExecutionPolicy>;
    auto __result_mid = _Copy()(__policy, __middle, std::move(__last), std::move(__out));
    if (__result_mid == nullopt)
      return nullopt;
    return _Copy()(__policy, std::move(__first), std::move(__middle), *std::move(__result_mid));
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_DEFAULTS_TRANSFORM_FAMILY_H
