// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wshift-count-negative %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wshift-count-negative %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wall %s

enum shiftof {
    X = (1<<-29) // expected-warning {{shift count is negative}}
};
