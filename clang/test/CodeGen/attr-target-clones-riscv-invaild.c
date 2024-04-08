// RUN: not %clang_cc1 -triple riscv64-linux-gnu -target-feature +i -S -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK-UNSUPPORT-EXT
// RUN: not %clang_cc1 -triple riscv64 -target-feature +i -S -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK-UNSUPPORT-OS

// CHECK-UNSUPPORT-EXT: error: Unsupport 'zicsr' for _riscv_hwprobe
__attribute__((target_clones("default", "arch=+zicsr"))) int foo1(void) {
  return 1;
}

// CHECK-UNSUPPORT-OS: error: Only Linux support _riscv_hwprobe
__attribute__((target_clones("default", "arch=+c"))) int foo2(void) {
  return 2;
}

int bar() { return foo1()+foo2(); }
