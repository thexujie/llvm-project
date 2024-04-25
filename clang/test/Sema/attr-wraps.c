// RUN: %clang_cc1 %s -verify -fsyntax-only -triple x86_64-pc-linux-gnu
// expected-no-diagnostics
typedef int __attribute__((wraps)) wrapping_int;
typedef unsigned __attribute__((wraps)) wrapping_u32;

int implicit_truncation(void) {
  const wrapping_int A = 1;
  return 2147483647 + A;
}

struct R {
  wrapping_int a: 2; // test bitfield sign change
  wrapping_u32 b: 1; // test bitfield overflow/truncation
};

void bitfields_truncation(void) {
  struct R r;
  r.a = 2; // this value changes from 2 to -2
  ++r.a;

  r.b = 2; // changes from 2 to 0
  ++r.b;
}
