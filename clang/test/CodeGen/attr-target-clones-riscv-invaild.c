// RUN: not %clang_cc1 -triple riscv64-linux-gnu -target-feature +i -S -emit-llvm -o - %s 2>&1 | FileCheck %s

// CHECK: error: Unsupport 'zicsr' for _riscv_hwprobe
__attribute__((target_clones("default", "arch=+zicsr"))) int foo1(void) {
  return 1;
}

int bar() { return foo1(); }
