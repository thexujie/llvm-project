// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Foo {};

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}

void test(int a, Foo b, void *c, int *d, Foo *e, const Foo *f, Incomplete *g) {
  __builtin_zero_non_value_bits(a); // expected-error {{passing 'int' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('int' vs structure pointer)}}
  __builtin_zero_non_value_bits(b); // expected-error {{passing 'Foo' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('Foo' vs structure pointer)}}
  __builtin_zero_non_value_bits(c); // expected-error {{passing 'void *' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('void *' vs structure pointer)}}
  __builtin_zero_non_value_bits(d); // expected-error {{passing 'int *' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('int *' vs structure pointer)}}
  __builtin_zero_non_value_bits(e); // This should not error.
  __builtin_zero_non_value_bits(f); // expected-error {{read-only variable is not assignable}}
  __builtin_zero_non_value_bits(g); // expected-error {{variable has incomplete type 'Incomplete'}}
}
