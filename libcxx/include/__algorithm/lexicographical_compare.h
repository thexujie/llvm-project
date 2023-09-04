//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
#define _LIBCPP___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/min.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__string/constexpr_c_functions.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_trivially_lexicographically_comparable.h>
#include <__type_traits/predicate_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter1, class _Sent1, class _Iter2, class _Sent2, class _Proj1, class _Proj2, class _Comp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 bool __lexicographical_compare(
    _Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Sent2 __last2, _Comp& __comp, _Proj1& __proj1, _Proj2& __proj2) {
  while (__first2 != __last2) {
    if (__first1 == __last1 ||
        std::__invoke(__comp, std::__invoke(__proj1, *__first1), std::__invoke(__proj2, *__first2)))
      return true;
    if (std::__invoke(__comp, std::__invoke(__proj2, *__first2), std::__invoke(__proj1, *__first1)))
      return false;
    ++__first1;
    ++__first2;
  }
  return false;
}

#if _LIBCPP_STD_VER >= 20
template <class _Tp, class _Up, class _Proj1, class _Proj2, class _Comp>
  requires(__libcpp_is_trivially_lexicographically_comparable<_Tp, _Up>::value && __is_identity<_Proj1>::value &&
           __is_identity<_Proj2>::value && __is_trivial_less_than_predicate<_Comp, _Tp, _Up>::value)
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 bool
__lexicographical_compare(_Tp* __first1, _Tp* __last1, _Up* __first2, _Up* __last2, _Comp&, _Proj1&, _Proj2&) {
  if (auto __res = std::__constexpr_memcmp(
          __first1, __first2, __element_count(std::min(__last1 - __first1, __last2 - __first2))))
    return __res < 0;
  return __last1 - __first1 < __last2 - __first2;
}
#endif

template <class _InputIterator1, class _InputIterator2, class _Compare>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 bool lexicographical_compare(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _Compare __comp) {
  __identity __proj;
  return std::__lexicographical_compare(
      std::__unwrap_iter(__first1),
      std::__unwrap_iter(__last1),
      std::__unwrap_iter(__first2),
      std::__unwrap_iter(__last2),
      __comp,
      __proj,
      __proj);
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_SINCE_CXX20 bool lexicographical_compare(
    _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return std::lexicographical_compare(__first1, __last1, __first2, __last2, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
