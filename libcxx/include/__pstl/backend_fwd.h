//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKEND_FWD_H
#define _LIBCPP___PSTL_BACKEND_FWD_H

#include <__config>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// find_if family
template <class _Backend, class _RawExecutionPolicy>
struct __find_if;

template <class _Backend, class _RawExecutionPolicy>
struct __find;

template <class _Backend, class _RawExecutionPolicy>
struct __find_if_not;

template <class _Backend, class _RawExecutionPolicy>
struct __any_of;

template <class _Backend, class _RawExecutionPolicy>
struct __all_of;

template <class _Backend, class _RawExecutionPolicy>
struct __none_of;

template <class _Backend, class _RawExecutionPolicy>
struct __is_partitioned;

// for_each family
template <class _Backend, class _RawExecutionPolicy>
struct __for_each;

template <class _Backend, class _RawExecutionPolicy>
struct __for_each_n;

template <class _Backend, class _RawExecutionPolicy>
struct __fill;

template <class _Backend, class _RawExecutionPolicy>
struct __fill_n;

template <class _Backend, class _RawExecutionPolicy>
struct __replace;

template <class _Backend, class _RawExecutionPolicy>
struct __replace_if;

template <class _Backend, class _RawExecutionPolicy>
struct __generate;

template <class _Backend, class _RawExecutionPolicy>
struct __generate_n;

// merge family
template <class _Backend, class _RawExecutionPolicy>
struct __merge;

// stable_sort family
template <class _Backend, class _RawExecutionPolicy>
struct __stable_sort;

template <class _Backend, class _RawExecutionPolicy>
struct __sort;

// transform family
template <class _Backend, class _RawExecutionPolicy>
struct __transform;

template <class _Backend, class _RawExecutionPolicy>
struct __transform_binary;

template <class _Backend, class _RawExecutionPolicy>
struct __replace_copy_if;

template <class _Backend, class _RawExecutionPolicy>
struct __replace_copy;

template <class _Backend, class _RawExecutionPolicy>
struct __move;

template <class _Backend, class _RawExecutionPolicy>
struct __copy;

template <class _Backend, class _RawExecutionPolicy>
struct __copy_n;

template <class _Backend, class _RawExecutionPolicy>
struct __rotate_copy;

// transform_reduce family
template <class _Backend, class _RawExecutionPolicy>
struct __transform_reduce;

template <class _Backend, class _RawExecutionPolicy>
struct __transform_reduce_binary;

template <class _Backend, class _RawExecutionPolicy>
struct __count_if;

template <class _Backend, class _RawExecutionPolicy>
struct __count;

template <class _Backend, class _RawExecutionPolicy>
struct __equal_3leg;

template <class _Backend, class _RawExecutionPolicy>
struct __equal;

template <class _Backend, class _RawExecutionPolicy>
struct __reduce;

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___PSTL_BACKEND_FWD_H
