//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_DEFAULTS_TRANSFORM_REDUCE_FAMILY_H
#define _LIBCPP___ALGORITHM_PSTL_DEFAULTS_TRANSFORM_REDUCE_FAMILY_H

#include <__algorithm/equal.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__pstl/backend_fwd.h>
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

// This header defines the default implementation for the transform_reduce family of algorithms:
// - count_if
// - count
// - equal(3 legs)
// - equal
// - reduce

template <class _Backend, class _RawExecutionPolicy>
struct __transform_reduce {
  template <class _Policy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp>
  operator()(_Policy&& __policy,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Tp const& __init,
             _BinaryOperation&& __reduce,
             _UnaryOperation&& __transform) const noexcept = delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __transform_reduce_binary {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _BinaryOperation1,
            class _BinaryOperation2>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp> operator()(
      _Policy&& __policy,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _Tp const& __init,
      _BinaryOperation1&& __reduce,
      _BinaryOperation2&& __transform) const noexcept = delete;
};

template <class _Backend, class _RawExecutionPolicy>
struct __count_if {
  template <class _Policy, class _ForwardIterator, class _Predicate>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__iter_diff_t<_ForwardIterator>> operator()(
      _Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate&& __pred) const noexcept {
    using _TransformReduce = __pstl::__transform_reduce<_Backend, _RawExecutionPolicy>;
    using _DiffT           = __iter_diff_t<_ForwardIterator>;
    using _Ref             = __iter_reference<_ForwardIterator>;
    return _TransformReduce()(
        __policy, std::move(__first), std::move(__last), _DiffT{}, std::plus{}, [&](_Ref __element) -> _DiffT {
          return __pred(__element) ? _DiffT(1) : _DiffT(0);
        });
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __count {
  template <class _Policy, class _ForwardIterator, class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<__iter_diff_t<_ForwardIterator>>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Tp const& __value) const noexcept {
    using _CountIf = __pstl::__count_if<_Backend, _RawExecutionPolicy>;
    using _Ref     = __iter_reference<_ForwardIterator>;
    return _CountIf()(__policy, std::move(__first), std::move(__last), [&](_Ref __element) -> bool {
      return __element == __value;
    });
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __equal_3leg {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&& __policy,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _Predicate&& __pred) const noexcept {
    using _TransformReduce = __pstl::__transform_reduce_binary<_Backend, _RawExecutionPolicy>;
    return _TransformReduce()(
        __policy,
        std::move(__first1),
        std::move(__last1),
        std::move(__first2),
        true,
        std::logical_and{},
        std::forward<_Predicate>(__pred));
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __equal {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<bool>
  operator()(_Policy&& __policy,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardIterator2 __last2,
             _Predicate&& __pred) const noexcept {
    if constexpr (__has_random_access_iterator_category<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category<_ForwardIterator2>::value) {
      if (__last1 - __first1 != __last2 - __first2)
        return false;
      // Fall back to the 3 legged algorithm
      using _Equal3Leg = __pstl::__equal_3leg<_Backend, _RawExecutionPolicy>;
      return _Equal3Leg()(
          __policy, std::move(__first1), std::move(__last1), std::move(__first2), std::forward<_Predicate>(__pred));
    } else {
      // If we don't have random access, fall back to the serial algorithm cause we can't do much
      return std::equal(
          std::move(__first1),
          std::move(__last1),
          std::move(__first2),
          std::move(__last2),
          std::forward<_Predicate>(__pred));
    }
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __reduce {
  template <class _Policy, class _ForwardIterator, class _Tp, class _BinaryOperation>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<_Tp>
  operator()(_Policy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation&& __op)
      const noexcept {
    using _TransformReduce = __pstl::__transform_reduce<_Backend, _RawExecutionPolicy>;
    return _TransformReduce()(
        __policy,
        std::move(__first),
        std::move(__last),
        std::move(__init),
        std::forward<_BinaryOperation>(__op),
        __identity{});
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_DEFAULTS_TRANSFORM_REDUCE_FAMILY_H
