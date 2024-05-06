// RUN: %clang -c --target=ppc32 -fcomplex-ppc-gnu-abi %s 2>&1

// RUN: not %clang %s --target=ppc64 -fcomplex-ppc-gnu-abi 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ERROR

// RUN: not %clang %s --target=ppc32-unknown-unknown-coff -fcomplex-ppc-gnu-abi 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ERROR

// CHECK-ERROR: error: unsupported option '-fcomplex-ppc-gnu-abi' for target
