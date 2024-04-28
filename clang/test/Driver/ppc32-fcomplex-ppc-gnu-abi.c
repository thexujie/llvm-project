// RUN: not %clang %s --target=x86_64 -fcomplex-ppc-gnu-abi 2>&1 \
// RUN:     | FileCheck %s -check-prefix=X86_64
// X86_64: error: unsupported option '-fcomplex-ppc-gnu-abi' for target 'x86_64'

// RUN: not %clang %s --target=ppc64 -fcomplex-ppc-gnu-abi 2>&1 \
// RUN:     | FileCheck %s -check-prefix=PPC64
// PPC64: error: unsupported option '-fcomplex-ppc-gnu-abi' for target 'ppc64'

// RUN: not %clang %s --target=riscv64 -fcomplex-ppc-gnu-abi 2>&1 \
// RUN:     | FileCheck %s -check-prefix=RISCV64
// RISCV64: error: unsupported option '-fcomplex-ppc-gnu-abi' for target 'riscv64'

// RUN: not %clang %s --target=aarch64 -fcomplex-ppc-gnu-abi 2>&1 \
// RUN:     | FileCheck %s -check-prefix=ARM64
// ARM64: error: unsupported option '-fcomplex-ppc-gnu-abi' for target 'aarch64'
