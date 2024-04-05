//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CONFIGURATION_H
#define _LIBCPP___PSTL_CONFIGURATION_H

#include <__config>

#if defined(_LIBCPP_PSTL_BACKEND_SERIAL)
#  include <__pstl/backends/serial.h>
#elif defined(_LIBCPP_PSTL_BACKEND_STD_THREAD)
#  include <__pstl/backends/std_thread.h>
#elif defined(_LIBCPP_PSTL_BACKEND_LIBDISPATCH)
#  include <__pstl/backends/libdispatch.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

#if defined(_LIBCPP_PSTL_BACKEND_SERIAL)
using __configured_backend = __serial_backend_tag;
#elif defined(_LIBCPP_PSTL_BACKEND_STD_THREAD)
using __configured_backend = __std_thread_backend_tag;
#elif defined(_LIBCPP_PSTL_BACKEND_LIBDISPATCH)
using __configured_backend = __libdispatch_backend_tag;
#else

// ...New vendors can add parallel backends here...

#  error "Invalid choice of a PSTL parallel backend"
#endif

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___PSTL_CONFIGURATION_H
