// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -I %t %t/A.cppm -emit-module-interface -o %t/A.pcm -verify
// RUN: %clang_cc1 -std=c++20 -I %t %t/B.cppm -emit-module-interface -o %t/B.pcm -verify
// RUN: %clang_cc1 -std=c++20 -I %t %t/C.cpp -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm -fsyntax-only -verify

//--- foo.h
namespace baz {
  using foo = char;
  using baz::foo;
}

//--- bar.h
class bar {
  bar(baz::foo);
};

//--- A.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module A;
export using AX = baz::foo;

//--- B.cppm
// expected-no-diagnostics
module;
#include "foo.h"
#include "bar.h"
export module B;
export using BX = baz::foo;
export using BY = bar;

//--- C.cpp
#include "foo.h"
import A;
#include "bar.h"
import B;
// Since modules are loaded lazily, force loading by performing a lookup.
using xxx = bar;
// expected-no-diagnostics
