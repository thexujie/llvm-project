//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_H

#include <__config>

/*
Combined OpenMP CPU and GPU Backend
===================================
Contrary to the CPU backends found in ./cpu_backends/, the OpenMP backend can
target both CPUs and GPUs. The OpenMP standard defines that when offloading code
to an accelerator, the compiler must generate a fallback code for execution on
the host. Thereby, the backend works as a CPU backend if no targeted accelerator
is available at execution time. The target regions can also be compiled directly
for a CPU architecture, for instance by adding the command-line option
`-fopenmp-targets=x86_64-pc-linux-gnu` in Clang.

Implicit Assumptions
--------------------
If the user provides a function pointer as an argument to a parallel algorithm,
it is assumed that it is the device pointer as there is currently no way to
check whether a host or device pointer was passed.

Mapping Clauses
---------------
In some of the parallel algorithms, the user is allowed to provide the same
iterator as input and output. Hence, the order of the maps matters. Therefore,
`pragma omp target data map(to:...)` must be used before
`pragma omp target data map(alloc:...)`. Conversely, the maps with map modifier
`release` must be placed before the maps with map modifier `from` when
transferring the result from the device to the host.

Exceptions
----------
Currently, GPU architectures do not handle exceptions. OpenMP target regions are
allowed to contain try/catch statements and throw expressions in Clang, but if a
throw expression is reached, it will terminate the program. That does not
conform with the C++ standard.

*/

#include <__algorithm/pstl_backends/openmp/backend.h>

#include <__algorithm/pstl_backends/openmp/any_of.h>
#include <__algorithm/pstl_backends/openmp/fill.h>
#include <__algorithm/pstl_backends/openmp/find_if.h>
#include <__algorithm/pstl_backends/openmp/for_each.h>
#include <__algorithm/pstl_backends/openmp/merge.h>
#include <__algorithm/pstl_backends/openmp/stable_sort.h>
#include <__algorithm/pstl_backends/openmp/transform.h>
#include <__algorithm/pstl_backends/openmp/transform_reduce.h>

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_H
