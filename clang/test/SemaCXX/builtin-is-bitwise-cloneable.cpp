// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Scalar types are bitwise clonable.
static_assert(__is_bitwise_cloneable(int));
static_assert(__is_bitwise_cloneable(int*));
// array
static_assert(__is_bitwise_cloneable(int[10]));

// non-scalar types.
static_assert(!__is_bitwise_cloneable(int&));


struct Forward; // expected-note 2{{forward declaration of 'Forward'}}
static_assert(!__is_bitwise_cloneable(Forward)); // expected-error {{incomplete type 'Forward' used in type trait expression}}

struct Foo { int a; };
static_assert(__is_bitwise_cloneable(Foo));

struct DynamicClass { virtual int Foo(); };
static_assert(__is_bitwise_cloneable(DynamicClass));

struct Bar { int& b; }; // trivially copyable
static_assert(__is_trivially_copyable(Bar));
static_assert(__is_bitwise_cloneable(Bar));

struct Bar2 { Bar2(const Bar2&); int& b; }; // non-trivially copyable
static_assert(!__is_trivially_copyable(Bar2));
static_assert(!__is_bitwise_cloneable(Bar2)); // int& non-scalar member.

struct DerivedBar2 : public Bar2 {};
static_assert(!__is_bitwise_cloneable(DerivedBar2)); // base Bar2 is non-bitwise-cloneable.


template <typename T>
void TemplateFunction() {
  static_assert(__is_bitwise_cloneable(T)); // expected-error {{incomplete type 'Forward' used in type trait expression}}
}
void CallTemplateFunc() {
  TemplateFunction<Forward>(); // expected-note {{in instantiation of function template specialization}}
  TemplateFunction<Foo>();
}
