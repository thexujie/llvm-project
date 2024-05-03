// RUN: clang-repl "int i = 10;" 'extern "C" int printf(const char*,...);' \
// RUN:            'auto r1 = printf("i = %d\n", i);' | FileCheck --check-prefix=CHECK-DRIVER %s
// UNSUPPORTED: system-aix
// CHECK-DRIVER: i = 10
// RUN: cat %s | clang-repl -Xcc -std=c++11 -Xcc -fno-delayed-template-parsing | FileCheck %s
extern "C" int printf(const char*,...);

char c = 'a';
c
// CHECK: (char) 'a'

struct NonPOD {        \
  static int sI;       \
  int I;               \
  NonPOD(): I(sI++) {} \
};
const char* PrintValueRuntime(const NonPOD* type) {             \
  switch (type->I) {                                            \
  default: return "out-of-bounds";                              \
  case 0: return "0"; case 1: return "1"; case 2: return "2";   \
  case 3: return "3"; case 4: return "4"; case 5: return "5";   \
  }                                                             \
}

int NonPOD::sI = 0;

NonPOD non_pod_arr[2][3];
// Check array order after the value printing transformation. Also make sure we
// can handle the forward declaration of operator new with placement.
non_pod_arr
// CHECK-NEXT: (NonPOD[2][3]) { { 0, 1, 2 }, { 3, 4, 5 } }

const char* c_str = "Goodbye, world!";
c_str
// CHECK-NEXT: (const char *) "Goodbye, world!"

const char* c_null_str = 0;
c_null_str
// CHECK-NEXT: (const char *) nullptr

"Hello, world"
// CHECK-NEXT: (const char[13]) "Hello, world"

int x = 42;
x
// CHECK-NEXT: (int) 42

&x
// CHECK-NEXT: (int *) @0x{{[0-9a-f]+}}

x - 2
// CHECK-NEXT: (int) 40

float f = 4.2f;
f
// CHECK-NEXT: (float) 4.20000f

double d = 4.21;
d
// CHECK-NEXT: (double) 4.21000000000

struct S1{};
S1 s1;
s1
// CHECK-NEXT: (S1 &) @0x{{[0-9a-f]+}}

S1{}
// CHECK-NEXT: (S1) @0x{{[0-9a-f]+}}

struct S2 {int d;} E = {22};
E
// CHECK-NEXT: (struct S2 &) @0x{{[0-9a-f]+}}
E.d
// CHECK-NEXT: (int) 22

struct S3 { int* p; S3() { p = new int(42); } ~S3() { delete p; } };
S3{}
// CHECK-NEXT: (S3) @0x{{[0-9a-f]+}}
S3 s3;
s3
// CHECK-NEXT: (S3 &) @0x{{[0-9a-f]+}}

struct S4 { ~S4() { printf("~S4()\n"); }};
S4{}
// CHECK-NEXT: (S4) @0x{{[0-9a-f]+}}

enum Enum{ e1 = -12, e2, e3=33, e4, e5 = 33};
e2
// CHECK-NEXT: (Enum) (e2) : int -11
::e1
// CHECK-NEXT: (Enum) (e1) : int -12

enum class Color { Black = 0, Red, Green };
Color::Black
// CHECK-NEXT: (Color) (Color::Black) : int 0

// Arrays.

int arr[3] = {1,2,3};
arr
// CHECK-NEXT: (int[3]) { 1, 2, 3 }


// Lambdas.

auto Lambda1 = []{};
Lambda1
// CHECK-NEXT: ((lambda) &) @0x{{[0-9a-f]+}}
[]{}
// CHECK-NEXT: ((lambda at input_line_{{[0-9]+}}:1:1)) @0x{{[0-9a-f]+}}

template<int n> struct F{ enum {RET=F<n-1>::RET*n} ; };
template<> struct F<0> { enum {RET = 1}; };
F<7>::RET
// CHECK-NEXT: (F<7>::(unnamed enum at input_line_{{[0-9]+}}:1:27)) (F<7>::RET) : unsigned int 5040

int foo() { return 42; }
foo()
// CHECK-NEXT: (int) 42

void bar() {}
bar()

struct S5 { int foo() { return 42; }};
&S5::foo
// CHECK-NEXT: (int (S5::*)()) Function @0x{{[0-9a-f]+}}

#include <memory>

auto p1 = std::make_shared<int>(42);
p1
// CHECK-NEXT: (std::shared_ptr<int> &) std::shared_ptr -> @0x{{[0-9a-f]+}}

std::unique_ptr<int> p2(new int(42));
p2
// CHECK-NEXT: (std::unique_ptr<int> &) std::unique_ptr -> @0x{{[0-9a-f]+}}

#include <array>
std::array<int, 3> a{1, 2, 3};
a
// CHECK-NEXT: (std::array<int, 3> &) { 1, 2, 3 }

#include <vector>
std::vector<int> v1 = {7, 5, 16, 8};
v1
// CHECK-NEXT: (std::vector<int> &) { 7, 5, 16, 8 }

std::vector<bool> v = {true, false, true};
v
// CHECK-NEXT: (std::vector<bool> &) { true, false, true }

#include <deque>
std::deque<int> dq = {7, 5, 16, 8};
dq
// CHECK-NEXT: (std::deque<int> &) { 7, 5, 16, 8 }

#include <forward_list>
std::forward_list<int> fl {3,4,5,6};
fl
// CHECK-NEXT: (std::forward_list<int> &) { 3, 4, 5, 6 }

#include <set>
std::set<int> z1 = {2,4,6,8};
z1
// CHECK-NEXT: (std::set<int> &) { 2, 4, 6, 8 }

#include <unordered_set>
std::unordered_set<int> z2 = {8,2,4,6};
z2
// CHECK-NEXT: (std::unordered_set<int> &) { [[Num:[0-9]+]], [[Num:[0-9]+]], [[Num:[0-9]+]], [[Num:[0-9]+]] }

std::multiset<int> e {3,2,1,2,4,7,3};
e
// CHECK-NEXT: (std::multiset<int> &) { 1, 2, 2, 3, 3, 4, 7 }

#include <string>
std::string std_str = "Hello, world!";
std_str
// CHECK-NEXT: (std::string &) "Hello, world!"

#include <utility>
std::pair<int,char> pr(42,'a');
pr
// CHECK-NEXT: (std::pair<int, char> &) { 42, 'a' }

#include <tuple>
std::tuple<int,double,char> tu(42,3.14,'a');
tu
// CHECK-NEXT: (std::tuple<int, double, char> &) { 42, 3.14000000000, 'a' }

#include <map>
std::map<const char*, int> m1{{"CPU", 10}, {"GPU", 15}, {"RAM", 20}};
m1
// CHECK-NEXT: (std::map<const char *, int> &) { "CPU" => 10, "GPU" => 15, "RAM" => 20 }

#include <unordered_map>
std::unordered_map<int, int> m2 = { {1,2}, {3,4}};
m2
// CHECK-NEXT: (std::unordered_map<int, int> &) { [[Num:[0-9]+]] => [[Num:[0-9]+]], [[Num:[0-9]+]] => [[Num:[0-9]+]] }

struct MyType {};
std::string PrintValueRuntime(const MyType*) { return "My pretty printer!"; }
MyType{}
// CHECK-NEXT: (MyType) My pretty printer!
%quit

