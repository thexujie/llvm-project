// RUN: %clang_cc1 %s -verify -fsyntax-only -triple x86_64-pc-linux-gnu
// expected-no-diagnostics
typedef int __attribute__((wraps)) wrapping_int;

int foo(void) {
  const wrapping_int A = 1;
  return 2147483647 + A;
}
