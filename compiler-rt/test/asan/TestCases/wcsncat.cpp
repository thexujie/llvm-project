// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK

#include <stdio.h>
#include <wchar.h>

int main() {
  wchar_t *start = L"X means ";
  wchar_t *append = L"dog";
  wchar_t goodDst[15];
  wcscpy(goodDst, start);
  wcsncat(goodDst, append, 5);

  wchar_t badDst[11];
  wcscpy(badDst, start);
  wcsncat(badDst, append, 1);
  printf("Good so far.\n");
  // CHECK: Good so far.
  wcsncat(badDst, append, 3); // Boom!
  // CHECK:ERROR: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]] at pc {{0x[0-9a-f]+}} bp {{0x[0-9a-f]+}} sp {{0x[0-9a-f]+}}
  // CHECK: WRITE of size {{[0-9]+}} at [[ADDR:0x[0-9a-f]+]] thread T0
  // CHECK: #0 [[ADDR:0x[0-9a-f]+]] in wcsncat {{.*}}\sanitizer_common_interceptors.inc:{{[0-9]+}}
  // CHECK: #1 [[ADDR:0x[0-9a-f]+]] in main {{.*}}\TestCases\wcsncat.cpp:23
  // CHECK: This frame has 2 object(s):
  // CHECK: HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
  // CHECK: (longjmp, SEH and C++ exceptions *are* supported)
  // CHECK: SUMMARY: AddressSanitizer: stack-buffer-overflow {{.*}} in main
  // CHECK: Shadow bytes around the buggy address:
  // CHECK: Shadow byte legend (one shadow byte represents 8 application bytes):
  // CHECK-NEXT: Addressable:           00
  // CHECK-NEXT: Partially addressable: 01 02 03 04 05 06 07
  // CHECK-NEXT: Heap left redzone:       fa
  // CHECK-NEXT: Freed heap region:       fd
  // CHECK-NEXT: Stack left redzone:      f1
  // CHECK-NEXT: Stack mid redzone:       f2
  // CHECK-NEXT: Stack right redzone:     f3
  // CHECK-NEXT: Stack after return:      f5
  // CHECK-NEXT: Stack use after scope:   f8
  // CHECK-NEXT: Global redzone:          f9
  // CHECK-NEXT: Global init order:       f6
  // CHECK-NEXT: Poisoned by user:        f7
  // CHECK-NEXT: Container overflow:      fc
  // CHECK-NEXT: Array cookie:            ac
  // CHECK-NEXT: Intra object redzone:    bb
  // CHECK-NEXT: ASan internal:           fe
  // CHECK-NEXT: Left alloca redzone:     ca
  // CHECK-NEXT: Right alloca redzone:    cb
}