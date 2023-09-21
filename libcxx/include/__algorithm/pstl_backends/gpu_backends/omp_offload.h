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
#include <__utility/move.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __par_backend {
inline namespace __omp_gpu_backend {

// In OpenMP, we need to extract the pointer for the underlying data for data
// structures like std::vector and std::array to be able to map the data to the
// device.

template <typename T>
_LIBCPP_HIDE_FROM_ABI inline T __omp_extract_base_ptr(T p) {
  return p;
}

template <typename T>
_LIBCPP_HIDE_FROM_ABI inline T __omp_extract_base_ptr(std::__wrap_iter<T> w) {
  std::pointer_traits<std::__wrap_iter<T>> PT;
  return PT.to_address(w);
}

// Applying function or lambda in a loop

template <class _Iterator, class _DifferenceType, class _Function>
_LIBCPP_HIDE_FROM_ABI _Iterator __omp_parallel_for_simd_1(_Iterator __first, _DifferenceType __n, _Function __f) noexcept {
  #pragma omp target teams distribute parallel for simd map(tofrom:__first[0:__n])
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
_LIBCPP_HIDE_FROM_ABI _Index __omp_parallel_for_simd_val_1(_Index __first, _DifferenceType __n, const _Tp& __value) noexcept {
  #pragma omp target teams distribute parallel for simd map(tofrom:__first[0:__n]) map(to:__value)
  for (_DifferenceType __i = 0; __i < __n; ++__i)
    __first[__i] = __value;

  return __first + __n;
}

template <class _Index, class _DifferenceType, class _Tp>
_LIBCPP_HIDE_FROM_ABI _Index __parallel_for_simd_val_1(_Index __first, _DifferenceType __n, const _Tp& __value) noexcept {
  __omp_parallel_for_simd_val_1(__omp_gpu_backend::__omp_extract_base_ptr(__first), __n, __value);
  return __first + __n;
}

} // namespace __omp_gpu_backend
} // namespace __par_backend

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && && _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKENDS_OMP_OFFLOAD_H
