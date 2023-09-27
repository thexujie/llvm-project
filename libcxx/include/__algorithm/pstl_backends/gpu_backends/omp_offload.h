//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_OMP_OFFLOAD_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_OMP_OFFLOAD_H

#include <__assert>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/wrap_iter.h>
#include <__memory/addressof.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_same.h>
#include <__utility/move.h>
#include <cstddef>

// is_same

// __libcpp_is_contiguous_iterator

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __par_backend {
inline namespace __omp_gpu_backend {

// Checking if a pointer is in a range
template <typename T1, typename T2, typename T3>
_LIBCPP_HIDE_FROM_ABI inline bool __omp_in_ptr_range(T1, T2, T3) {
  return false;
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI inline bool __omp_in_ptr_range(_Tp* a, _Tp* p, _Tp* b) {
  return std::less_equal<_Tp*>{}(a, p) && std::less<_Tp*>{}(p, b);
}

// In OpenMP, we need to extract the pointer for the underlying data for data
// structures like std::vector and std::array to be able to map the data to the
// device.

template <typename _Tp, std::enable_if<std::is_pointer<_Tp>::value >::type* = 0>
_LIBCPP_HIDE_FROM_ABI inline _Tp __omp_extract_base_ptr(_Tp p) {
  return p;
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI inline auto __omp_extract_base_ptr(_Tp p) {
  return std::addressof(*p);
  ;
}

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI inline _Tp __omp_extract_base_ptr(std::__wrap_iter<_Tp> w) {
  std::pointer_traits<std::__wrap_iter<_Tp>> PT;
  return PT.to_address(w);
}

//===----------------------------------------------------------------------===//
// Templates for one iterator
//===----------------------------------------------------------------------===//

// Applying function or lambda in a loop

template <class _Iterator, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator __omp_parallel_for_simd_1(
    _Iterator __first, _DifferenceType __n, _Function __f, [[maybe_unused]] const int __device = 0) noexcept {
#  pragma omp target teams distribute parallel for simd map(tofrom : __first[0 : __n]) device(__device)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __f(__first[__i]);

  return __first + __n;
}

// Extracting the underlying pointer

template <class _Iterator, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator __parallel_for_simd_1(_Iterator __first, _DifferenceType __n, _Function __f) noexcept {
  __omp_parallel_for_simd_1(__omp_gpu_backend::__omp_extract_base_ptr(__first), __n, __f);
  return __first + __n;
}

// Assigning a value in a loop

template <class _Index, class _DifferenceType, class _Tp>
_LIBCPP_HIDE_FROM_ABI _Index __omp_parallel_for_simd_val_1(
    _Index __first, _DifferenceType __n, const _Tp& __value, [[maybe_unused]] const int __device = 0) noexcept {
#  pragma omp target teams distribute parallel for simd map(from : __first[0 : __n]) map(always, to : __value)         \
      device(__device)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __first[__i] = __value;

  return __first + __n;
}

template <class _Index, class _DifferenceType, class _Tp>
_LIBCPP_HIDE_FROM_ABI _Index
__parallel_for_simd_val_1(_Index __first, _DifferenceType __n, const _Tp& __value) noexcept {
  __omp_parallel_for_simd_val_1(__omp_gpu_backend::__omp_extract_base_ptr(__first), __n, __value);
  return __first + __n;
}

//===----------------------------------------------------------------------===//
// Templates for two iterators
//===----------------------------------------------------------------------===//

template <class _Iterator1, class _DifferenceType, class _Iterator2, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator1 __omp_parallel_for_simd_2(
    _Iterator1 __first1,
    _DifferenceType __n,
    _Iterator2 __first2,
    _Function __f,
    [[maybe_unused]] const int __device = 0) noexcept {
  if ((!std::is_same<_Iterator1, _Iterator2>::value) ||
      (std::is_same<_Iterator1, _Iterator2>::value &&
       !__omp_gpu_backend::__omp_in_ptr_range(__first1, __first2, __first1 + __n))) {
#  pragma omp target teams distribute parallel for simd map(to : __first1[0 : __n]) map(from : __first2[0 : __n])      \
      device(__device)
    for (_DifferenceType __i = 0; __i < __n; ++__i)
      __first2[__i] = __f(__first1[__i]);
    return __first1 + __n;
  }
#  pragma omp target teams distribute parallel for simd map(tofrom : __first1[0 : __n], __first2[0 : __n])             \
      device(__device)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __first2[__i] = __f(__first1[__i]);

  return __first1 + __n;
}

// Extracting the underlying pointer

template <class _Iterator1, class _DifferenceType, class _Iterator2, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator1
__parallel_for_simd_2(_Iterator1 __first1, _DifferenceType __n, _Iterator2 __first2, _Function __f) noexcept {
  __omp_parallel_for_simd_2(
      __omp_gpu_backend::__omp_extract_base_ptr(__first1),
      __n,
      __omp_gpu_backend::__omp_extract_base_ptr(__first2),
      __f);
  return __first1 + __n;
}

//===----------------------------------------------------------------------===//
// Templates for three iterator
//===----------------------------------------------------------------------===//

template <class _Iterator1, class _DifferenceType, class _Iterator2, class _Iterator3, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator1 __omp_parallel_for_simd_3(
    _Iterator1 __first1,
    _DifferenceType __n,
    _Iterator2 __first2,
    _Iterator3 __first3,
    _Function __f,
    [[maybe_unused]] const int __device = 0) noexcept {
  // It may be that __first3 is in the interval [__first1+__n) or [__firt2+__n)
  // It is, however, undefined behavior to compare two pointers that do not
  // point to the same object or are not the same type.
  // If we can prove that __first3 is not in any of the ranges [__first1+__n)
  // or [__firt2+__n), it is safe to reduce the amount of data copied to and
  // from the device
  constexpr bool are_not_same_type =
      !std::is_same<_Iterator1, _Iterator2>::value && !std::is_same<_Iterator1, _Iterator3>::value;
  const bool no_overlap_13 =
      std::is_same<_Iterator1, _Iterator3>::value &&
      !__omp_gpu_backend::__omp_in_ptr_range(__first1, __first3, __first1 + __n);
  const bool no_overlap_23 =
      std::is_same<_Iterator2, _Iterator3>::value &&
      !__omp_gpu_backend::__omp_in_ptr_range(__first2, __first3, __first2 + __n);
  if (are_not_same_type || (no_overlap_13 && no_overlap_23)) {
#  pragma omp target teams distribute parallel for simd map(to : __first1[0 : __n], __first2[0 : __n])                 \
      map(from : __first3[0 : __n]) device(__device)
    for (_DifferenceType __i = 0; __i < __n; ++__i)
      __first3[__i] = __f(__first1[__i], __first2[__i]);
    return __first1 + __n;
  }
  // In the general case, we have to map all data to and from the device
#  pragma omp target teams distribute parallel for simd map(                                                           \
          tofrom : __first1[0 : __n], __first2[0 : __n], __first3[0 : __n]) device(__device)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __first3[__i] = __f(__first1[__i], __first2[__i]);

  return __first1 + __n;
}

// Extracting the underlying pointer

template <class _Iterator1, class _DifferenceType, class _Iterator2, class _Iterator3, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator1 __parallel_for_simd_3(
    _Iterator1 __first1, _DifferenceType __n, _Iterator2 __first2, _Iterator3 __first3, _Function __f) noexcept {
  __omp_parallel_for_simd_3(
      __omp_gpu_backend::__omp_extract_base_ptr(__first1),
      __n,
      __omp_gpu_backend::__omp_extract_base_ptr(__first2),
      __omp_gpu_backend::__omp_extract_base_ptr(__first3),
      __f);
  return __first1 + __n;
}

//===----------------------------------------------------------------------===//
// Templates for reductions
//===----------------------------------------------------------------------===//

// General case

#  define __PSTL_OMP_SIMD_1_REDUCTION(omp_op, std_op)                                                                                 \
    template <class _Iterator,                                                                                                   \
              class _DifferenceType,                                                                                             \
              typename _Tp,                                                                                                      \
              typename _BinaryOperationType,                                                                                     \
              typename _UnaryOperation>                                                                     \
    _LIBCPP_HIDE_FROM_ABI _Tp __omp_parallel_for_simd_reduction_1(                                                               \
        _Iterator __first,                                                                                                       \
        _DifferenceType __n,                                                                                                     \
        _Tp __init,                                                                                                              \
        std_op<_BinaryOperationType> __reduce,                                                                                   \
        _UnaryOperation __transform/*,                                                                                             \
        [[maybe_unused]] const int __device = 0*/) noexcept {    \
_PSTL_PRAGMA(omp target teams distribute parallel for simd reduction(omp_op:__init) map(to : __first[0 : __n])) /*device(__device))*/ \
      for (_DifferenceType __i = 0; __i < __n; ++__i)                                                                                 \
        __init = __reduce(__init, __transform(__first[__i]));                                                                         \
      return __init;                                                                                                                  \
    }

#  define __PSTL_OMP_SIMD_2_REDUCTION(omp_op, std_op)                                                                                                     \
    template <class _Iterator1,                                                                                                                      \
              class _Iterator2,                                                                                                                      \
              class _DifferenceType,                                                                                                                 \
              typename _Tp,                                                                                                                          \
              typename _BinaryOperationType,                                                                                                         \
              typename _UnaryOperation >                                                                                         \
    _LIBCPP_HIDE_FROM_ABI _Tp __omp_parallel_for_simd_reduction_2(                                                                                   \
        _Iterator1 __first1,                                                                                                                         \
        _Iterator2 __first2,                                                                                                                         \
        _DifferenceType __n,                                                                                                                         \
        _Tp __init,                                                                                                                                  \
        std_op<_BinaryOperationType> __reduce,                                                                                                       \
        _UnaryOperation __transform/*,                                                                                                                 \
        [[maybe_unused]] const int __device = 0*/) noexcept {    \
_PSTL_PRAGMA(omp target teams distribute parallel for simd reduction(omp_op:__init) map(to : __first1[0 : __n], __first2[0 : __n]))/* device(__device))*/ \
      for (_DifferenceType __i = 0; __i < __n; ++__i)                                                                                                     \
        __init = __reduce(__init, __transform(__first1[__i], __first2[__i]));                                                                             \
      return __init;                                                                                                                                      \
    }

#  define __PSTL_OMP_SIMD_REDUCTION(omp_op, std_op)                                                                    \
    __PSTL_OMP_SIMD_1_REDUCTION(omp_op, std_op)                                                                        \
    __PSTL_OMP_SIMD_2_REDUCTION(omp_op, std_op)

// Addition
__PSTL_OMP_SIMD_REDUCTION(+, std::plus)

// Subtraction
__PSTL_OMP_SIMD_REDUCTION(-, std::minus)

// Multiplication
__PSTL_OMP_SIMD_REDUCTION(*, std::multiplies)

// Logical and
__PSTL_OMP_SIMD_REDUCTION(&&, std::logical_and)

// Logical or
__PSTL_OMP_SIMD_REDUCTION(||, std::logical_or)

// Bitwise and
__PSTL_OMP_SIMD_REDUCTION(&, std::bit_and)

// Bitwise or
__PSTL_OMP_SIMD_REDUCTION(|, std::bit_or)

// Bitwise xor
__PSTL_OMP_SIMD_REDUCTION(^, std::bit_xor)

// Extracting the underlying pointers

template <class _Iterator, class _DifferenceType, typename _Tp, typename _BinaryOperation, typename _UnaryOperation >
_LIBCPP_HIDE_FROM_ABI _Tp __parallel_for_simd_reduction_1(
    _Iterator __first,
    _DifferenceType __n,
    _Tp __init,
    _BinaryOperation __reduce,
    _UnaryOperation __transform,
    [[maybe_unused]] const int __device = 0) noexcept {
  return __omp_parallel_for_simd_reduction_1(
      __omp_gpu_backend::__omp_extract_base_ptr(__first), __n, __init, __reduce, __transform);
}

template <class _Iterator1,
          class _Iterator2,
          class _DifferenceType,
          typename _Tp,
          typename _BinaryOperation,
          typename _UnaryOperation >
_LIBCPP_HIDE_FROM_ABI _Tp __parallel_for_simd_reduction_2(
    _Iterator1 __first1,
    _Iterator2 __first2,
    _DifferenceType __n,
    _Tp __init,
    _BinaryOperation __reduce,
    _UnaryOperation __transform,
    [[maybe_unused]] const int __device = 0) noexcept {
  return __omp_parallel_for_simd_reduction_2(
      __omp_gpu_backend::__omp_extract_base_ptr(__first1),
      __omp_gpu_backend::__omp_extract_base_ptr(__first2),
      __n,
      __init,
      __reduce,
      __transform);
}

} // namespace __omp_gpu_backend
} // namespace __par_backend

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_OMP_OFFLOAD_H
