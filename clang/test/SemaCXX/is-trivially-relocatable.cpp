// RUN: %clang_cc1 -std=c++03 -fsyntax-only -verify %s -triple x86_64-windows-msvc
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s -triple x86_64-windows-msvc
// RUN: %clang_cc1 -std=c++03 -fsyntax-only -verify %s -triple x86_64-apple-darwin10
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s -triple x86_64-apple-darwin10

// expected-no-diagnostics

#if __cplusplus < 201103L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__, "")
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

template <class T>
struct Agg {
  T t_;
};

template <class T>
struct Der : T {
};

template <class T>
struct Mut {
  mutable T t_;
};

template <class T>
struct Non {
  Non(); // make it a non-aggregate
  T t_;
};

struct CompletelyTrivial {
};
static_assert(__is_trivially_relocatable(CompletelyTrivial));
static_assert(__is_trivially_relocatable(Agg<CompletelyTrivial>));
static_assert(__is_trivially_relocatable(Der<CompletelyTrivial>));
static_assert(__is_trivially_relocatable(Mut<CompletelyTrivial>));
static_assert(__is_trivially_relocatable(Non<CompletelyTrivial>));

struct NonTrivialDtor {
  ~NonTrivialDtor();
};
static_assert(!__is_trivially_relocatable(NonTrivialDtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialDtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialDtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialDtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialDtor>));

struct NonTrivialCopyCtor {
  NonTrivialCopyCtor(const NonTrivialCopyCtor&);
};
static_assert(!__is_trivially_relocatable(NonTrivialCopyCtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialCopyCtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialCopyCtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialCopyCtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialCopyCtor>));

struct NonTrivialMutableCopyCtor {
  NonTrivialMutableCopyCtor(NonTrivialMutableCopyCtor&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMutableCopyCtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMutableCopyCtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMutableCopyCtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMutableCopyCtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMutableCopyCtor>));

#if __cplusplus >= 201103L
struct NonTrivialMoveCtor {
  NonTrivialMoveCtor(NonTrivialMoveCtor&&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMoveCtor));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMoveCtor>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMoveCtor>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMoveCtor>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMoveCtor>));
#endif

struct NonTrivialCopyAssign {
  NonTrivialCopyAssign& operator=(const NonTrivialCopyAssign&);
};
static_assert(!__is_trivially_relocatable(NonTrivialCopyAssign));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialCopyAssign>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialCopyAssign>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialCopyAssign>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialCopyAssign>));

struct NonTrivialMutableCopyAssign {
  NonTrivialMutableCopyAssign& operator=(NonTrivialMutableCopyAssign&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMutableCopyAssign));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMutableCopyAssign>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMutableCopyAssign>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMutableCopyAssign>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMutableCopyAssign>));

#if __cplusplus >= 201103L
struct NonTrivialMoveAssign {
  NonTrivialMoveAssign& operator=(NonTrivialMoveAssign&&);
};
static_assert(!__is_trivially_relocatable(NonTrivialMoveAssign));
static_assert(!__is_trivially_relocatable(Agg<NonTrivialMoveAssign>));
static_assert(!__is_trivially_relocatable(Der<NonTrivialMoveAssign>));
static_assert(!__is_trivially_relocatable(Mut<NonTrivialMoveAssign>));
static_assert(!__is_trivially_relocatable(Non<NonTrivialMoveAssign>));
#endif

#if __cplusplus >= 202002L
template<bool B>
struct EligibleNonTrivialDefaultCtor {
    EligibleNonTrivialDefaultCtor() requires B;
    EligibleNonTrivialDefaultCtor() = default;
};
// Only the Rule of 5 members (not default ctor) affect trivial relocatability.
static_assert(__is_trivially_relocatable(EligibleNonTrivialDefaultCtor<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialDefaultCtor<false>));

template<bool B>
struct IneligibleNonTrivialDefaultCtor {
    IneligibleNonTrivialDefaultCtor();
    IneligibleNonTrivialDefaultCtor() requires B = default;
};
// Only the Rule of 5 members (not default ctor) affect trivial relocatability.
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDefaultCtor<true>));
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDefaultCtor<false>));

template<bool B>
struct EligibleNonTrivialCopyCtor {
    EligibleNonTrivialCopyCtor(const EligibleNonTrivialCopyCtor&) requires B;
    EligibleNonTrivialCopyCtor(const EligibleNonTrivialCopyCtor&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialCopyCtor<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialCopyCtor<false>));

template<bool B>
struct IneligibleNonTrivialCopyCtor {
    IneligibleNonTrivialCopyCtor(const IneligibleNonTrivialCopyCtor&);
    IneligibleNonTrivialCopyCtor(const IneligibleNonTrivialCopyCtor&) requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialCopyCtor<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialCopyCtor<false>));

template<bool B>
struct EligibleNonTrivialMoveCtor {
    EligibleNonTrivialMoveCtor(EligibleNonTrivialMoveCtor&&) requires B;
    EligibleNonTrivialMoveCtor(EligibleNonTrivialMoveCtor&&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialMoveCtor<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialMoveCtor<false>));

template<bool B>
struct IneligibleNonTrivialMoveCtor {
    IneligibleNonTrivialMoveCtor(IneligibleNonTrivialMoveCtor&&);
    IneligibleNonTrivialMoveCtor(IneligibleNonTrivialMoveCtor&&) requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialMoveCtor<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialMoveCtor<false>));

template<bool B>
struct EligibleNonTrivialCopyAssign {
    EligibleNonTrivialCopyAssign& operator=(const EligibleNonTrivialCopyAssign&) requires B;
    EligibleNonTrivialCopyAssign& operator=(const EligibleNonTrivialCopyAssign&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialCopyAssign<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialCopyAssign<false>));

template<bool B>
struct IneligibleNonTrivialCopyAssign {
    IneligibleNonTrivialCopyAssign& operator=(const IneligibleNonTrivialCopyAssign&);
    IneligibleNonTrivialCopyAssign& operator=(const IneligibleNonTrivialCopyAssign&) requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialCopyAssign<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialCopyAssign<false>));

template<bool B>
struct EligibleNonTrivialMoveAssign {
    EligibleNonTrivialMoveAssign& operator=(EligibleNonTrivialMoveAssign&&) requires B;
    EligibleNonTrivialMoveAssign& operator=(EligibleNonTrivialMoveAssign&&) = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialMoveAssign<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialMoveAssign<false>));

template<bool B>
struct IneligibleNonTrivialMoveAssign {
    IneligibleNonTrivialMoveAssign& operator=(IneligibleNonTrivialMoveAssign&&);
    IneligibleNonTrivialMoveAssign& operator=(IneligibleNonTrivialMoveAssign&&) requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialMoveAssign<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialMoveAssign<false>));

template<bool B>
struct EligibleNonTrivialDtor {
    ~EligibleNonTrivialDtor() requires B;
    ~EligibleNonTrivialDtor() = default;
};
static_assert(!__is_trivially_relocatable(EligibleNonTrivialDtor<true>));
static_assert(__is_trivially_relocatable(EligibleNonTrivialDtor<false>));

template<bool B>
struct IneligibleNonTrivialDtor {
    ~IneligibleNonTrivialDtor();
    ~IneligibleNonTrivialDtor() requires B = default;
};
static_assert(__is_trivially_relocatable(IneligibleNonTrivialDtor<true>));
static_assert(!__is_trivially_relocatable(IneligibleNonTrivialDtor<false>));
#endif
