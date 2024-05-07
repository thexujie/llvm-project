// RUN: %clang_cc1 -fsyntax-only -verify=expected,beforeCxx2b -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2b -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 -Wmissing-format-attribute %s

typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

namespace std
{
    template<class Elem> struct basic_string_view {};
    template<class Elem> struct basic_string {
        const Elem *c_str() const noexcept;
        basic_string(const basic_string_view<Elem> SW);
    };

    using string = basic_string<char>;
    using wstring = basic_string<wchar_t>;
    using string_view = basic_string_view<char>;
    using wstring_view = basic_string_view<wchar_t>;
}

__attribute__((__format__(__printf__, 1, 2)))
int printf(const char *, ...); // #printf

__attribute__((__format__(__scanf__, 1, 2)))
int scanf(const char *, ...); // #scanf

__attribute__((__format__(__printf__, 1, 0)))
int vprintf(const char *, va_list); // #vprintf

__attribute__((__format__(__scanf__, 1, 0)))
int vscanf(const char *, va_list); // #vscanf

__attribute__((__format__(__printf__, 2, 0)))
int vsprintf(char *, const char *, va_list); // #vsprintf

__attribute__((__format__(__printf__, 3, 0)))
int vsnprintf(char *ch, size_t, const char *, va_list); // #vsnprintf

void f1(const std::string &str, ... /* args */) // #f1
{
    va_list args;
    vscanf(str.c_str(), args); // no warning
    vprintf(str.c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f2(const std::string &str, ... /* args */); // #f2

void f3(std::string_view str, ... /* args */) // #f3
{
    va_list args;
    vscanf(std::string(str).c_str(), args); // no warning
    vprintf(std::string(str).c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f4(std::string_view str, ... /* args */); // #f4

void f5(const std::wstring &str, ... /* args */) // #f5
{
    va_list args;
    vscanf((const char *)str.c_str(), args); // no warning
    vprintf((const char *)str.c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f6(const std::wstring &str, ... /* args */); // #f6

void f7(std::wstring_view str, ... /* args */) // #f7
{
    va_list args;
    vscanf((const char *) std::wstring(str).c_str(), args); // no warning
    vprintf((const char *) std::wstring(str).c_str(), args); // no warning
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f8(std::wstring_view str, ... /* args */); // #f8

void f9(const wchar_t *out, ... /* args */) // #f9
{
    va_list args;
    vprintf(out, args); // expected-error {{no matching function for call to 'vprintf'}}
                        // expected-note@#vprintf {{candidate function not viable: no known conversion from 'const wchar_t *' to 'const char *' for 1st argument}}
    vscanf((const char *) out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f9'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((char *) out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f9'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f10(const wchar_t *out, ... /* args */); // #f10

void f11(const char16_t *out, ... /* args */) // #f11
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
                       // expected-note@#vscanf {{candidate function not viable: no known conversion from 'const char16_t *' to 'const char *' for 1st argument}}
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f12(const char16_t *out, ... /* args */); // #f12

void f13(const char32_t *out, ... /* args */) // #f13
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
                       // expected-note@#vscanf {{candidate function not viable: no known conversion from 'const char32_t *' to 'const char *' for 1st argument}}
}

__attribute__((format(scanf, 1, 2))) // expected-error {{format argument not a string type}}
void f14(const char32_t *out, ... /* args */); // #f14

void f15(const char *out, ... /* args */) // #f15
{
    va_list args;
    vscanf(out, args); // expected-warning@#f15 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f15'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(scanf, 1, 2)))
void f16(const char *out, ... /* args */) // #f16
{
    va_list args;
    vscanf(out, args); // no warning
}

void f17(const unsigned char *out, ... /* args */) // #f17
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
                       // expected-note@#vscanf {{candidate function not viable: no known conversion from 'const unsigned char *' to 'const char *' for 1st argument}}
}

__attribute__((format(scanf, 1, 2)))
void f18(const unsigned char *out, ... /* args */) // #f18
{
    va_list args;
    vprintf(out, args); // expected-error {{no matching function for call to 'vprintf'}}
                        // expected-note@#vprintf {{candidate function not viable: no known conversion from 'const unsigned char *' to 'const char *' for 1st argument}}
}

void f19(const signed char *out, ... /* args */) // #f19
{
    va_list args;
    vprintf(out, args); // expected-error {{no matching function for call to 'vprintf'}}
                        // expected-note@#vprintf {{candidate function not viable: no known conversion from 'const signed char *' to 'const char *' for 1st argument}}
}

__attribute__((format(scanf, 1, 2)))
void f20(const signed char *out, ... /* args */) // #f20
{
    va_list args;
    vscanf(out, args); // expected-error {{no matching function for call to 'vscanf'}}
                       // expected-note@#vscanf {{candidate function not viable: no known conversion from 'const signed char *' to 'const char *' for 1st argument}}
}

void f21(const char out[], ... /* args */) // #f21
{
    va_list args;
    vscanf(out, args); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f21'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(scanf, 1, 0)))
void f22(const char out[], ... /* args */) // #f22
{
    va_list args;
    vscanf(out, args); // no warning
}

void f23(const char *out) // #f23
{
    va_list args;
    vscanf(out, args); // expected-warning@#f23 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f23'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f24(const char *out, va_list args) // #f24
{
    vprintf(out, args); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

typedef va_list tdVaList;
typedef int tdInt;
void f25(const char *out, ... /* args */) // #f25
{
    tdVaList args;
    printf(out, args); // expected-warning@#f25 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f25'}}
                       // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    tdInt a;
    scanf(out, a); // expected-warning@#f25 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f25'}}
                   // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f26(const char *out, tdVaList args) // #f26
{
    scanf(out, args); // expected-warning@#f26 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f26'}}
                      // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
    tdInt a;
    printf(out, a); // expected-warning@#f26 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f26'}}
                    // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

struct S1
{
    void fn1(const char *out, ... /* args */) // #S1_fn1
    {
        va_list args;
        vscanf(out, args); // expected-warning@#S1_fn1 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn1'}}
                           // CHECK-FIXES: __attribute__((format(scanf, 2, 3)))
    }

    __attribute__((format(scanf, 2, 0)))
    void fn2(const char *out, va_list args); // #S1_fn2

    void fn3(const char *out, ... /* args */);

    void fn4(this S1& expliciteThis, const char *out, va_list args) // #S1_fn4
    {
        expliciteThis.fn2(out, args); // beforeCxx2b-error@#S1_fn4 {{explicit object parameters are incompatible with C++ standards before C++2b}}
                                      // expected-warning@#S1_fn4 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn4'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 2, 0)))
    }
};

void S1::fn3(const char *out, ... /* args */) // #S1_fn3
{
    va_list args;
    fn2(out, args); // expected-warning@#S1_fn3 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn3'}}
                    // CHECK-FIXES: __attribute__((format(scanf, 2, 3)))
}

union U1
{
    __attribute__((format(printf, 2, 0)))
    void fn1(const char *out, va_list args); // #U1_fn1

    void fn2(const char *out, ... /* args */) // #U1_fn2
    {
        va_list args;
        fn1(out, args); // expected-warning@#U1_fn2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn2'}}
                        // CHECK-FIXES: __attribute__((format(printf, 2, 3)))
    }

    void fn3(this U1&, const char *out) // #U1_fn3
    {
        va_list args;
        printf(out, args); // beforeCxx2b-error@#U1_fn3 {{explicit object parameters are incompatible with C++ standards before C++2b}}
                           // expected-warning@#U1_fn3 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn3'}}
                           // CHECK-FIXES: __attribute__((format(printf, 2, 0)))
    }
};

class C1
{
    __attribute__((format(printf, 3, 0)))
    void fn1(const int n, const char *out, va_list args); // #C1_fn1

    void fn2(const char *out, const int n, ... /* args */) // #C1_fn2
    {
        va_list args;
        fn1(n, out, args); // expected-warning@#C1_fn2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'fn2'}}
                           // CHECK-FIXES: __attribute__((format(printf, 2, 4)))
    }

    void fn3(this const C1&, const char *out, va_list args) // #C1_fn3
    {
        scanf(out, args); // beforeCxx2b-error@#C1_fn3 {{explicit object parameters are incompatible with C++ standards before C++2b}}
                          // expected-warning@#C1_fn3 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'fn3'}}
                          // CHECK-FIXES: __attribute__((format(scanf, 2, 0)))
    }

    C1(const int n, const char *out) //#C1_C1a
    {
        va_list args;
        fn1(n, out, args); // expected-warning@#C1_C1a {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'C1'}}
                           // CHECK-FIXES: __attribute__((format(printf, 3, 0)))
    }

    C1(const char *out, ... /* args */) // #C1_C1b
    {
        va_list args;
        printf(out, args); // expected-warning@#C1_C1b {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'C1'}}
                           // CHECK-FIXES: __attribute__((format(printf, 2, 3)))
    }

    ~C1()
    {
        const char *out;
        va_list args;
        vprintf(out, args); // no warning
    }
};

// TODO: implement for templates
template <int N>
void func(char (&str)[N], ... /* args */)
{
    va_list args;
    vprintf(str, args); // no warning
}
