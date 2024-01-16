// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s

#if __cplusplus >= 202002L

/// FIXME: This should work, but currently doesn't.
/// The global variable is visited twice, and we don't currently
/// handle failure when initializing a global variable. So the
/// second time we try to use already freed memory.
#if 0
constexpr int *a = new int(12);
#endif


constexpr int a() {
  new int(12); // expected-note {{allocation performed here was not deallocated}} \
               // ref-note {{allocation performed here was not deallocated}}
  return 1;
}
static_assert(a() == 1, ""); // expected-error {{not an integral constant expression}} \
                             // ref-error {{not an integral constant expression}}

constexpr int b() {
  int *i = new int(12);
  int m = *i;
  delete(i);
  return m;
}
static_assert(b() == 12, "");


struct S {
  int a;
  int b;

  static constexpr S *create(int a, int b) {
    return new S(a, b);
  }
};

constexpr int c() {
  S *s = new S(12, 13);

  int i = s->a;
  delete s;

  return i;
}
static_assert(c() == 12, "");

/// Dynamic allocation in function ::create(), freed in function d().
constexpr int d() {
  S* s = S::create(12, 14);

  int sum = s->a + s->b;
  delete s;
  return sum;
}
static_assert(d() == 26);


/// Test we emit the right diagnostic for several allocations done on
/// the same site.
constexpr int loop() {
  for (int i = 0; i < 10; ++i) {
    int *a = new int[10]; // ref-note {{not deallocated (along with 9 other memory leaks)}} \
                          // expected-note {{not deallocated (along with 9 other memory leaks)}}
  }

  return 1;
}
static_assert(loop() == 1, ""); // ref-error {{not an integral constant expression}} \
                                // expected-error {{not an integral constant expression}}


/// No initializer.
constexpr int noInit() {
  int *i = new int;
  delete i;
  return 0;
}
static_assert(noInit() == 0, "");

/// Try to delete a pointer that hasn't been heap allocated.
/// FIXME: pretty-printing the pointer is broken here.
constexpr int notHeapAllocated() { // ref-error {{never produces a constant expression}} \
                                   // expected-error {{never produces a constant expression}}
  int A = 0; // ref-note 2{{declared here}} \
             // expected-note 2{{declared here}}
  delete &A; // ref-note 2{{delete of pointer '&A' that does not point to a heap-allocated object}} \
             // expected-note 2{{delete of pointer '' that does not point to a heap-allocated object}}

  return 1;
}
static_assert(notHeapAllocated() == 1, ""); // ref-error {{not an integral constant expression}} \
                                            // ref-note {{in call to 'notHeapAllocated()'}} \
                                            // expected-error {{not an integral constant expression}} \
                                            // expected-note {{in call to 'notHeapAllocated()'}}

consteval int deleteNull() {
  int *A = nullptr;
  delete A;
  return 1;
}
static_assert(deleteNull() == 1, "");

consteval int doubleDelete() { // ref-error {{never produces a constant expression}} \
                               // expected-error {{never produces a constant expression}}
  int *A = new int;
  delete A;
  delete A; // ref-note 2{{delete of pointer that has already been deleted}} \
            // expected-note 2{{delete of pointer that has already been deleted}}
  return 1;
}
static_assert(doubleDelete() == 1); // ref-error {{not an integral constant expression}} \
                                    // ref-note {{in call to 'doubleDelete()'}} \
                                    // expected-error {{not an integral constant expression}} \
                                    // expected-note {{in call to 'doubleDelete()'}}

consteval int largeArray1(bool b) {
  if (b) {
    int *a = new int[1ull<<32]; // ref-note {{cannot allocate array; evaluated array bound 4294967296 is too large}} \
                                // expected-note {{cannot allocate array; evaluated array bound 4294967296 is too large}}
    delete[] a;
  }
  return 1;
}
static_assert(largeArray1(false) == 1, "");
static_assert(largeArray1(true) == 1, ""); // ref-error {{not an integral constant expression}} \
                                           // ref-note {{in call to 'largeArray1(true)'}} \
                                           // expected-error {{not an integral constant expression}}  \
                                           // expected-note {{in call to 'largeArray1(true)'}}

consteval int largeArray2(bool b) {
  if (b) {
    S *a = new S[1ull<<32]; // ref-note {{cannot allocate array; evaluated array bound 4294967296 is too large}} \
                            // expected-note {{cannot allocate array; evaluated array bound 4294967296 is too large}}
    delete[] a;
  }
  return 1;
}
static_assert(largeArray2(false) == 1, "");
static_assert(largeArray2(true) == 1, ""); // ref-error {{not an integral constant expression}} \
                                           // ref-note {{in call to 'largeArray2(true)'}} \
                                           // expected-error {{not an integral constant expression}}  \
                                           // expected-note {{in call to 'largeArray2(true)'}}

namespace Arrays {
  constexpr int d() {
    int *Arr = new int[12];

    Arr[0] = 1;
    Arr[1] = 5;

    int sum = Arr[0] + Arr[1];
    delete[] Arr;
    return sum;
  }
  static_assert(d() == 6);


  constexpr int mismatch1() { // ref-error {{never produces a constant expression}} \
                              // expected-error {{never produces a constant expression}}
    int *i = new int(12); // ref-note {{allocated with 'new' here}} \
                          // ref-note 2{{heap allocation performed here}} \
                          // expected-note {{allocated with 'new' here}} \
                          // expected-note 2{{heap allocation performed here}}
    delete[] i; // ref-warning {{'delete[]' applied to a pointer that was allocated with 'new'}} \
                // ref-note 2{{array delete used to delete pointer to non-array object of type 'int'}} \
                // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'}} \
                // expected-note 2{{array delete used to delete pointer to non-array object of type 'int'}}
    return 6;
  }
  static_assert(mismatch1() == 6); // ref-error {{not an integral constant expression}} \
                                   // ref-note {{in call to 'mismatch1()'}} \
                                   // expected-error {{not an integral constant expression}} \
                                   // expected-note {{in call to 'mismatch1()'}}


  constexpr int mismatch2() { // ref-error {{never produces a constant expression}} \
                              // expected-error {{never produces a constant expression}}
    int *i = new int[12]; // ref-note {{allocated with 'new[]' here}} \
                          // ref-note 2{{heap allocation performed here}} \
                          // expected-note {{allocated with 'new[]' here}} \
                          // expected-note 2{{heap allocation performed here}}
    delete i; // ref-warning {{'delete' applied to a pointer that was allocated with 'new[]'}} \
              // ref-note 2{{non-array delete used to delete pointer to array object of type 'int[12]'}} \
              // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'}} \
              // expected-note 2{{non-array delete used to delete pointer to array object of type 'int[12]'}}
    return 6;
  }
  static_assert(mismatch2() == 6); // ref-error {{not an integral constant expression}} \
                                   // ref-note {{in call to 'mismatch2()'}} \
                                   // expected-error {{not an integral constant expression}} \
                                   // expected-note {{in call to 'mismatch2()'}}
  /// Array of composite elements.
  constexpr int foo() {
    S *ss = new S[12];

    ss[0].a = 12;

    int m = ss[0].a;

    delete[] ss;
    return m;
  }
  static_assert(foo() == 12);
}

/// From test/SemaCXX/cxx2a-consteval.cpp

namespace std {
template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T &> { using type = T; };
template <typename T> struct remove_reference<T &&> { using type = T; };
template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T &&t) noexcept {
  return static_cast<typename std::remove_reference<T>::type &&>(t);
}
}

namespace cxx2a {
struct A {
  int* p = new int(42); // expected-note 7{{heap allocation performed here}} \
                        // ref-note 7{{heap allocation performed here}}
  consteval int ret_i() const { return p ? *p : 0; }
  consteval A ret_a() const { return A{}; }
  constexpr ~A() { delete p; }
};

consteval int by_value_a(A a) { return a.ret_i(); }

consteval int const_a_ref(const A &a) {
  return a.ret_i();
}

consteval int rvalue_ref(const A &&a) {
  return a.ret_i();
}

consteval const A &to_lvalue_ref(const A &&a) {
  return a;
}

void test() {
  constexpr A a{ nullptr };
  { int k = A().ret_i(); }

  { A k = A().ret_a(); } // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                         // expected-note {{heap-allocated object is not a constant expression}} \
                         // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                         // ref-note {{heap-allocated object is not a constant expression}}
  { A k = to_lvalue_ref(A()); } // expected-error {{'cxx2a::to_lvalue_ref' is not a constant expression}} \
                                // expected-note {{reference to temporary is not a constant expression}} \
                                // expected-note {{temporary created here}} \
                                // ref-error {{'cxx2a::to_lvalue_ref' is not a constant expression}} \
                                // ref-note {{reference to temporary is not a constant expression}} \
                                // ref-note {{temporary created here}}
  { A k = to_lvalue_ref(A().ret_a()); } // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                        // expected-note {{heap-allocated object is not a constant expression}} \
                                        // expected-error {{'cxx2a::to_lvalue_ref' is not a constant expression}} \
                                        // expected-note {{reference to temporary is not a constant expression}} \
                                        // expected-note {{temporary created here}} \
                                        // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                        // ref-note {{heap-allocated object is not a constant expression}} \
                                        // ref-error {{'cxx2a::to_lvalue_ref' is not a constant expression}} \
                                        // ref-note {{reference to temporary is not a constant expression}} \
                                        // ref-note {{temporary created here}}
  { int k = A().ret_a().ret_i(); } // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                   // expected-note {{heap-allocated object is not a constant expression}} \
                                   // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                   // ref-note {{heap-allocated object is not a constant expression}}
  { int k = by_value_a(A()); }
  { int k = const_a_ref(A()); }
  { int k = const_a_ref(a); }
  { int k = rvalue_ref(A()); }
  { int k = rvalue_ref(std::move(a)); }
  { int k = const_a_ref(A().ret_a()); } // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                        // expected-note {{is not a constant expression}} \
                                        // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                        // ref-note {{is not a constant expression}}


  { int k = const_a_ref(to_lvalue_ref(A().ret_a())); } // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                                       // expected-note {{is not a constant expression}} \
                                                       // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                                       // ref-note {{is not a constant expression}}
  { int k = const_a_ref(to_lvalue_ref(std::move(a))); }
  { int k = by_value_a(A().ret_a()); }
  { int k = by_value_a(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = (A().ret_a(), A().ret_i()); } // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                          // expected-note {{is not a constant expression}} \
                                          // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                          // ref-note {{is not a constant expression}} \
                                          // expected-warning {{left operand of comma operator has no effect}} \
                                          // ref-warning {{left operand of comma operator has no effect}}
  { int k = (const_a_ref(A().ret_a()), A().ret_i()); }  // expected-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                                        // expected-note {{is not a constant expression}} \
                                                        // ref-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                                        // ref-note {{is not a constant expression}} \
                                                        // expected-warning {{left operand of comma operator has no effect}} \
                                                        // ref-warning {{left operand of comma operator has no effect}}
}
}

#else
/// Make sure we reject this prior to C++20
constexpr int a() { // ref-error {{never produces a constant expression}} \
                    // expected-error {{never produces a constant expression}}
  delete new int(12); // ref-note 2{{dynamic memory allocation is not permitted in constant expressions until C++20}} \
                      // expected-note 2{{dynamic memory allocation is not permitted in constant expressions until C++20}}
  return 1;
}
static_assert(a() == 1, ""); // ref-error {{not an integral constant expression}} \
                             // ref-note {{in call to 'a()'}} \
                             // expected-error {{not an integral constant expression}} \
                             // expected-note {{in call to 'a()'}}
#endif
