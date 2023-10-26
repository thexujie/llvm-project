// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wshift-negative-value %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wshift-negative-value %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wall %s

enum shiftof {
    X = (-1<<29) // expected-warning {{shifting a negative signed value is undefined}}
};
