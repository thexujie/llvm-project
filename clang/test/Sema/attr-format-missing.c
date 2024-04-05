// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s

typedef unsigned short char16_t;
typedef unsigned int char32_t;
typedef __WCHAR_TYPE__ wchar_t;
typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

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

__attribute__((__format__(__scanf__, 1, 4)))
void f1(char *out, const size_t len, const char *format, ... /* args */) // #f1
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning@#f1 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f1'}}
                                       // CHECK-FIXES: __attribute__((format(printf, 3, 4)))
}

__attribute__((__format__(__printf__, 1, 4)))
void f2(char *out, const size_t len, const char *format, ... /* args */) // #f2
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning@#f2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f2'}}
                                       // CHECK-FIXES: __attribute__((format(printf, 3, 4)))
}

void f3(char *out, va_list args) // #f3
{
    vprintf(out, args); // expected-warning@#f3 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f3'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
    vscanf(out, args); // expected-warning@#f3 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f3'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f4(char* out, ... /* args */) // #f4
{
    va_list args;
    vprintf("test", args); // no warning

    const char *ch;
    vscanf(ch, args); // no warning
}

void f5(va_list args) // #f5
{
    char *ch;
    vscanf(ch, args); // no warning
}

void f6(char *out, va_list args) // #f6
{
    char *ch;
    vscanf(ch, args); // no warning
    vprintf("test", args); // no warning
}

void f7(const char *out, ... /* args */) // #f7
{
    va_list args;

    vscanf(out, &args[0]); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f7'}}
                           // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
    vprintf(out, &args[0]); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f7'}}
                            // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

__attribute__((format(scanf, 1, 0)))
__attribute__((format(printf, 1, 2)))
void f8(const char *out, ... /* args */) // #f8
{
    va_list args;

    vscanf(out, &args[0]); // no warning
    vprintf(out, &args[0]); // no warning
}

void f9(const char out[], ... /* args */) // #f9
{
    va_list args;
    char *ch;
    vscanf(ch, args); // no warning
    vsprintf(ch, out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f9'}}
                             // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
}

void f10(const wchar_t *out, ... /* args */) // #f10
{
    va_list args;
    vprintf(out, args);
#if __SIZEOF_WCHAR_T__ == 4
                        // expected-warning@-2 {{incompatible pointer types passing 'const wchar_t *' (aka 'const int *') to parameter of type 'const char *'}}
#else
                        // expected-warning@-4 {{incompatible pointer types passing 'const wchar_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
#endif
                        // expected-note@#vprintf {{passing argument to parameter here}}
                        // expected-warning@#f10 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f10'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((const char *) out, args); // expected-warning@#f10 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f10'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((char *) out, args); // expected-warning@#f10 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f10'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f11(const wchar_t *out, ... /* args */); // #f11

void f12(const char16_t *out, ... /* args */) // #f12
{
    va_list args;
    vscanf(out, args); // expected-warning {{incompatible pointer types passing 'const char16_t *' (aka 'const unsigned short *') to parameter of type 'const char *'}}
                       // expected-note@#vscanf {{passing argument to parameter here}}
                       // expected-warning@#f12 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f12'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2))) // expected-error {{format argument not a string type}}
void f13(const char16_t *out, ... /* args */); // #f13

void f14(const char32_t *out, ... /* args */) // #f14
{
    va_list args;
    vscanf(out, args); // expected-warning {{incompatible pointer types passing 'const char32_t *' (aka 'const unsigned int *') to parameter of type 'const char *'}}
                       // expected-note@#vscanf {{passing argument to parameter here}}
                       // expected-warning@#f14 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f14'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(scanf, 1, 2))) // expected-error {{format argument not a string type}}
void f15(const char32_t *out, ... /* args */); // #f15

void f16(const unsigned char *out, ... /* args */) // #f16
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // expected-note@#vprintf {{passing argument to parameter here}}
                        // expected-warning@#f16 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f16'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((const char *) out, args); // no warning
                                      // expected-warning@#f16 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((char *) out, args); // no warning
                                // expected-warning@#f16 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f16'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

__attribute__((format(printf, 1, 2)))
void f17(const unsigned char *out, ... /* args */) // #f17
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'const unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // expected-note@#vprintf {{passing argument to parameter here}}
    vscanf((const char *) out, args); // expected-warning@#f17 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f17'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vprintf((const char *) out, args); // no warning
    vscanf((char *) out, args); // expected-warning@#f17 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f17'}}
                                // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vprintf((char *) out, args); // no warning
}

void f18(signed char *out, ... /* args */) // #f18
{
    va_list args;
    vscanf(out, args); // expected-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}} \
                       // expected-note@#vscanf {{passing argument to parameter here}} \
                       // expected-warning@#f18 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vscanf((const char *) out, args); // expected-warning@#f18 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f18'}}
                                      // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
    vprintf((char *) out, args); // expected-warning@#f18 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f18'}}
                                 // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
}

__attribute__((format(scanf, 1, 2)))
void f19(signed char *out, ... /* args */) // #f19
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'signed char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // expected-note@#vprintf {{passing argument to parameter here}}
                        // expected-warning@#f19 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                        // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((const char *) out, args); // no warning
    vprintf((const char *) out, args); // expected-warning@#f19 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                                       // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
    vscanf((char *) out, args); // no warning
    vprintf((char *) out, args); // expected-warning@#f19 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f19'}}
                                 // CHECK-FIXES: __attribute__((format(printf, 1, 2)))
}

__attribute__((format(printf, 1, 2)))
void f20(unsigned char out[], ... /* args */) // #f20
{
    va_list args;
    vprintf(out, args); // expected-warning {{passing 'unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                        // expected-note@#vprintf {{passing argument to parameter here}}
    vscanf(out, args); // expected-warning {{passing 'unsigned char *' to parameter of type 'const char *' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
                       // expected-note@#vscanf {{passing argument to parameter here}}
                       // expected-warning@#f20 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f20'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 2)))
}

void f21(char* out) // #f21
{
    va_list args;
    const char* ch;
    vsprintf(out, ch, args); // no warning
    vscanf(out, args); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f21'}}
                       // CHECK-FIXES: __attribute__((format(scanf, 1, 0)))
}

void f22(const char *out, ... /* args */) // #f22
{
    int a;
    printf(out, a); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                    // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
    printf(out, 1); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                    // CHECK-FIXES: __attribute__((format(printf, 1, 0)))
}

__attribute__((format(printf, 1, 2)))
void f23(const char *out, ... /* args */) // #f23
{
    int a;
    printf(out, a); // no warning
    printf(out, 1); // no warning
}

void f24(char* ch, const char *out, ... /* args */) // #f24
{
    va_list args;
    printf(ch, args); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24}}
                      // CHECK-FIXES: __attribute__((format(printf, 1, 3)))
    int a;
    printf(out, a); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                    // CHECK-FIXES: __attribute__((format(printf, 2, 0)))
    printf(out, 1); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                    // CHECK-FIXES: __attribute__((format(printf, 2, 0)))
    printf(out, args); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f24'}}
                       // CHECK-FIXES: __attribute__((format(printf, 2, 3)))
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
