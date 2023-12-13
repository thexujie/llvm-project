// RUN: mkdir -p %t
// RUN: %clang++ %s -o %t/run
// RUN: %t/run

#include <cassert>
#include <cstdio>
#include <cstring>
#include <new>

template <class T>
void print_bytes(const T *object)
{
  auto size = sizeof(T);
  const unsigned char * const bytes = reinterpret_cast<const unsigned char *>(object);
  size_t i;

  fprintf(stderr, "[ ");
  for(i = 0; i < size; i++)
  {
    fprintf(stderr, "%02x ", bytes[i]);
  }
  fprintf(stderr, "]\n");
}

template <size_t A1, size_t A2, class T>
struct alignas(A1) BasicWithPadding {
  T x;
  alignas(A2) T y;
};

template <size_t A1, size_t A2, size_t N, class T>
struct alignas(A1) SpacedArrayMembers {
  T x[N];
  alignas(A2) char c;
  T y[N];
};

template <size_t A1, size_t A2, class T>
struct alignas(A1) PaddedPointerMembers {
  T *x;
  alignas(A2) T *y;
};

template <size_t A1, size_t A2, size_t A3, class T>
struct alignas(A1) ThreeMembers {
  T x;
  alignas(A2) T y;
  alignas(A3) T z;
};

template <class T>
struct Normal {
  T a;
  T b;
};

template <class T>
struct X {
  T x;
};

template <class T>
struct Z {
  T z;
};

template <size_t A, class T>
struct YZ : public Z<T> {
  alignas(A) T y;
};

template <size_t A1, size_t A2, class T>
struct alignas(A1) HasBase : public X<T>, public YZ<A2, T> {
  T a;
  alignas(A2) T b;
};

template <size_t A1, size_t A2, class T>
void testAllForType(T a, T b, T c, T d) {
  using B = BasicWithPadding<A1, A2, T>;
  B basic1;
  memset(&basic1, 0, sizeof(B));
  basic1.x = a;
  basic1.y = b;
  B basic2;
  memset(&basic2, 42, sizeof(B));
  basic2.x = a;
  basic2.y = b;
  assert(memcmp(&basic1, &basic2, sizeof(B)) != 0);
  __builtin_zero_non_value_bits(&basic2);
  assert(memcmp(&basic1, &basic2, sizeof(B)) == 0);
  using A = SpacedArrayMembers<A1, A2, 2, T>;
  A arr1;
  memset(&arr1, 0, sizeof(A));
  arr1.x[0] = a;
  arr1.x[1] = b;
  arr1.y[0] = c;
  arr1.y[1] = d;
  A arr2;
  memset(&arr2, 42, sizeof(A));
  arr2.x[0] = a;
  arr2.x[1] = b;
  arr2.y[0] = c;
  arr2.y[1] = d;
  arr2.c = 0;
  assert(memcmp(&arr1, &arr2, sizeof(A)) != 0);
  __builtin_zero_non_value_bits(&arr2);
  assert(memcmp(&arr1, &arr2, sizeof(A)) == 0);

  using P = PaddedPointerMembers<A1, A2, T>;
  P ptr1;
  memset(&ptr1, 0, sizeof(P));
  ptr1.x = &a;
  ptr1.y = &b;
  P ptr2;
  memset(&ptr2, 42, sizeof(P));
  ptr2.x = &a;
  ptr2.y = &b;
  assert(memcmp(&ptr1, &ptr2, sizeof(P)) != 0);
  __builtin_zero_non_value_bits(&ptr2);
  assert(memcmp(&ptr1, &ptr2, sizeof(P)) == 0);

  using Three = ThreeMembers<A1, A2, A2, T>;
  Three three1;
  memset(&three1, 0, sizeof(Three));
  three1.x = a;
  three1.y = b;
  three1.z = c;
  Three three2;
  memset(&three2, 42, sizeof(Three));
  three2.x = a;
  three2.y = b;
  three2.z = c;
  __builtin_zero_non_value_bits(&three2);
  assert(memcmp(&three1, &three2, sizeof(Three)) == 0);

  using N = Normal<T>;
  N normal1;
  memset(&normal1, 0, sizeof(N));
  normal1.a = a;
  normal1.b = b;
  N normal2;
  memset(&normal2, 42, sizeof(N));
  normal2.a = a;
  normal2.b = b;
  __builtin_zero_non_value_bits(&normal2);
  assert(memcmp(&normal1, &normal2, sizeof(N)) == 0);

  using H = HasBase<A1, A2, T>;
  H base1;
  memset(&base1, 0, sizeof(H));
  base1.a = a;
  base1.b = b;
  base1.x = c;
  base1.y = d;
  base1.z = a;
  H base2;
  memset(&base2, 42, sizeof(H));
  base2.a = a;
  base2.b = b;
  base2.x = c;
  base2.y = d;
  base2.z = a;
  assert(memcmp(&base1, &base2, sizeof(H)) != 0);
  __builtin_zero_non_value_bits(&base2);
  unsigned i = 0;
  assert(memcmp(&base1, &base2, sizeof(H)) == 0);
}

struct UnsizedTail {
  int size;
  alignas(8) char buf[];

  UnsizedTail(int size) : size(size) {}
};

void otherTests() {
  const size_t size1 = sizeof(UnsizedTail) + 4;
  char buff1[size1];
  char buff2[size1];
  memset(buff1, 0, size1);
  memset(buff2, 42, size1);
  auto *u1 = new (buff1) UnsizedTail(4);
  u1->buf[0] = 1;
  u1->buf[1] = 2;
  u1->buf[2] = 3;
  u1->buf[3] = 4;
  auto *u2 = new (buff2) UnsizedTail(4);
  u2->buf[0] = 1;
  u2->buf[1] = 2;
  u2->buf[2] = 3;
  u2->buf[3] = 4;
  assert(memcmp(u1, u2, sizeof(UnsizedTail)) != 0);
  __builtin_zero_non_value_bits(u2);
  assert(memcmp(u1, u2, sizeof(UnsizedTail)) == 0);

  using B = BasicWithPadding<8, 4, char>;
  auto *basic1 = new B;
  memset(basic1, 0, sizeof(B));
  basic1->x = 1;
  basic1->y = 2;
  auto *basic2 = new B;
  memset(basic2, 42, sizeof(B));
  basic2->x = 1;
  basic2->y = 2;
  assert(memcmp(basic1, basic2, sizeof(B)) != 0);
  __builtin_zero_non_value_bits(basic2);
  assert(memcmp(basic1, basic2, sizeof(B)) == 0);
  delete basic2;
  delete basic1;

  using B = BasicWithPadding<8, 4, char>;
  B *basic3 = new B;
  memset(basic3, 0, sizeof(B));
  basic3->x = 1;
  basic3->y = 2;
  B *basic4 = new B;
  memset(basic4, 42, sizeof(B));
  basic4->x = 1;
  basic4->y = 2;
  assert(memcmp(basic3, basic4, sizeof(B)) != 0);
  __builtin_zero_non_value_bits(const_cast<volatile B *>(basic4));
  assert(memcmp(basic3, basic4, sizeof(B)) == 0);
  delete basic4;
  delete basic3;
}

struct Foo {
  int x;
  int y;
};

typedef float Float4Vec __attribute__((ext_vector_type(4)));
typedef float Float3Vec __attribute__((ext_vector_type(3)));

struct S1 {
 int x = 0;
 char c = 0;
};

struct S2{
  [[no_unique_address]] S1 s1;
  bool b;
  long double l;
  bool b2;
};

struct S3{
  [[no_unique_address]] S1 s1;
  bool b;
};

struct alignas(32) S4 {
  int i;
};
struct B1{

};

struct B2 {
  int x;
};
struct B3{
  char c;
};

struct B4{
  bool b;
};

struct B5{
  int x2;
};

struct D:B1,B2,B3,B4,B5{
  long double l;
  bool b2;
};


int main() {
  /*
  S2 s2{};

  memset(&s2, 42, sizeof(S2));
  s2.s1.x = 0x12345678;
  s2.s1.c = 0xff;
  s2.b = true;
  s2.l = 3.333;
  s2.b2 = true;
  print_bytes(&s2);
  __builtin_zero_non_value_bits(&s2);
  print_bytes(&s2);

  D s2{};
  memset(&s2, 42, sizeof(D));
  s2.x = 0x12345678;
  s2.c = 0xff;
  s2.b = true;
  s2.x2 = 0x87654321;
  s2.l = 3.333;
  s2.b2 = true;
  print_bytes(&s2);
  __builtin_zero_non_value_bits(&s2);
  print_bytes(&s2);

  S3 s2[2];

  memset(&s2, 42, 2*sizeof(S3));
  s2[0].s1.x = 0x12345678;
  s2[0].s1.c = 0xff;
  s2[0].b = true;
  s2[1].s1.x = 0x12345678;
  s2[1].s1.c = 0xff;
  s2[1].b = true;
  print_bytes(&s2);
  __builtin_zero_non_value_bits(&s2);
  print_bytes(&s2);

  */
/*
  S4 s2[2];

  memset(&s2, 42, 2*sizeof(S4));
  s2[0].i = 0x12345678;
  s2[1].i = 0x12345678;
  print_bytes(&s2);
  __builtin_zero_non_value_bits(&s2);
  print_bytes(&s2);


  assert(false);
*/
  testAllForType<32, 16, char>(11, 22, 33, 44);
  testAllForType<64, 32, char>(4, 5, 6, 7);
  testAllForType<32, 16, volatile char>(11, 22, 33, 44);
  testAllForType<64, 32, volatile char>(4, 5, 6, 7);
  testAllForType<32, 16, int>(0, 1, 2, 3);
  testAllForType<64, 32, int>(4, 5, 6, 7);
  testAllForType<32, 16, volatile int>(0, 1, 2, 3);
  testAllForType<64, 32, volatile int>(4, 5, 6, 7);
  testAllForType<32, 16, double>(0, 1, 2, 3);
  testAllForType<64, 32, double>(4, 5, 6, 7);
  testAllForType<32, 16, _ExtInt(28)>(0, 1, 2, 3);
  testAllForType<64, 32, _ExtInt(28)>(4, 5, 6, 7);
  testAllForType<32, 16, _ExtInt(60)>(0, 1, 2, 3);
  testAllForType<64, 32, _ExtInt(60)>(4, 5, 6, 7);
  testAllForType<32, 16, _ExtInt(64)>(0, 1, 2, 3);
  testAllForType<64, 32, _ExtInt(64)>(4, 5, 6, 7);
  testAllForType<32, 16, Foo>(Foo{1, 2}, Foo{3, 4}, Foo{1, 2}, Foo{3, 4});
  testAllForType<64, 32, Foo>(Foo{1, 2}, Foo{3, 4}, Foo{1, 2}, Foo{3, 4});
  testAllForType<256, 128, Float3Vec>(0, 1, 2, 3);
  testAllForType<128, 128, Float3Vec>(4, 5, 6, 7);
  testAllForType<256, 128, Float4Vec>(0, 1, 2, 3);
  testAllForType<128, 128, Float4Vec>(4, 5, 6, 7);

  otherTests();
  return 0;
}
