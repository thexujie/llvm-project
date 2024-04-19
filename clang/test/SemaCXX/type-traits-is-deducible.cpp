// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s 
// expected-no-diagnostics

template<typename T>
struct Foo {};
static_assert(__is_deducible(Foo, Foo<int>));
static_assert(!__is_deducible(Foo, int));

template <class T>
using AFoo1 = Foo<T*>;
static_assert(__is_deducible(AFoo1, Foo<int*>));
static_assert(!__is_deducible(AFoo1, Foo<int>));

template <class T>
using AFoo2 = Foo<int>;
static_assert(!__is_deducible(AFoo2, Foo<int>));

// default template argument counts.
template <class T = double>
using AFoo3 = Foo<int>;
static_assert(__is_deducible(AFoo3, Foo<int>));


template <int N>
struct Bar { int k = N; };
static_assert(__is_deducible(Bar, Bar<1>));

template <int N>
using ABar1 = Bar<N>;
static_assert(__is_deducible(ABar1, Bar<3>));
template <int N>
using ABar2 = Bar<1>;
static_assert(!__is_deducible(ABar2, Bar<1>));


template <typename T>
class Forward;
static_assert(__is_deducible(Forward, Forward<int>));
template <typename T>
using AForward = Forward<T>;
static_assert(__is_deducible(AForward, Forward<int>));


template <class T, T N>
using AArrary = int[N];
static_assert (__is_deducible(AArrary, int[42]));
static_assert (!__is_deducible(AArrary, double[42]));
