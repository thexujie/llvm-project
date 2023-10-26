// RUN: %clang_cc1 -fsyntax-only -verify -Wshift-count-overflow %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

enum shiftof {
    X = (1<<32) // expected-warning {{shift count >= width of type}}
};
