// RUN: %clangxx_nsan -O2 -g -DFN=StrFry %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRFRY %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrSep %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRSEP %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrTok %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRTOK %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrDup %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRDUP %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrNDup %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRNDUP %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StpCpy %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STPCPY %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrCpy %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRCPY %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrNCpy %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRNCPY %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrCat %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRCAT %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DFN=StrNCat %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=STRNCAT %s < %t.out

// This test case checks libc string operations interception.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "helpers.h"

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes, size_t bytes_per_line, size_t reserved);

void StrFry(char* const s) {
  strfry(s);
  __nsan_dump_shadow_mem(s, sizeof(float), sizeof(float), 0);
// strfry just destroys the whole area.
// STRFRY: StrFry
// STRFRY-NEXT: f0 f1 f2 f3
// STRFRY-NEXT: __ __ __ f3
}

void StrSep(char* const s) {
  char* sc = s;
  strsep(&sc, "\x40");
  __nsan_dump_shadow_mem(s, sizeof(float), sizeof(float), 0);
// strsep destroys the element that was replaced with a null character.
// STRSEP: StrSep
// STRSEP-NEXT: f0 f1 f2 f3
// STRSEP-NEXT: f0 __ f2 f3
}

void StrTok(char* const s) {
  strtok(s, "\x40");
  __nsan_dump_shadow_mem(s, sizeof(float), sizeof(float), 0);
// strtok just destroys the whole area except the terminator.
// STRTOK: StrTok
// STRTOK-NEXT: f0 f1 f2 f3
// STRTOK-NEXT: __ __ __ f3
}

void StrDup(char* const s) {
  char* const dup = strdup(s);
  __nsan_dump_shadow_mem(dup, 4, 4, 0);
  free(dup);
// STRDUP: StrDup
// STRDUP-NEXT: f0 f1 f2 f3
// STRDUP-NEXT: f0 f1 f2 __
}


void StrNDup(char* const s) {
  char* const dup = strndup(s, 2);
  __nsan_dump_shadow_mem(dup, 3, 3, 0);
  free(dup);
// STRNDUP: StrNDup
// STRNDUP-NEXT: f0 f1 f2 f3
// STRNDUP-NEXT: f0 f1 __
}

void StpCpy(char* const s) {
  char buffer[] = "abcdef\0";
  stpcpy(buffer, s);
  __nsan_dump_shadow_mem(buffer, sizeof(buffer), sizeof(buffer), 0);
// STPCPY: StpCpy
// STPCPY-NEXT: f0 f1 f2 f3
// STPCPY-NEXT: f0 f1 f2 __
}

void StrCpy(char* const s) {
  char buffer[] = "abcdef\0";
  strcpy(buffer, s);
  __nsan_dump_shadow_mem(buffer, sizeof(buffer), sizeof(buffer), 0);
// STRCPY: StrCpy
// STRCPY-NEXT: f0 f1 f2 f3
// STRCPY-NEXT: f0 f1 f2 __
}

void StrNCpy(char* const s) {
  char buffer[] = "abcdef\0";
  strncpy(buffer, s, 2);
  __nsan_dump_shadow_mem(buffer, sizeof(buffer), sizeof(buffer), 0);
// STRNCPY: StrNCpy
// STRNCPY-NEXT: f0 f1 f2 f3
// STRNCPY-NEXT: f0 f1 __
}

void StrCat(char* const s) {
  char buffer[] = "abcd\0    ";
  strcat(buffer, s);
  __nsan_dump_shadow_mem(buffer, sizeof(buffer), sizeof(buffer), 0);
// STRCAT: StrCat
// STRCAT-NEXT: f0 f1 f2 f3
// STRCAT-NEXT: __ __ __ __ f0 f1 f2 __
}

void StrNCat(char* const s) {
  char buffer[] = "abcd\0    ";
  strncat(buffer, s, 2);
  __nsan_dump_shadow_mem(buffer, sizeof(buffer), sizeof(buffer), 0);
// STRNCAT: StrNCat
// STRNCAT-NEXT: f0 f1 f2 f3
// STRNCAT-NEXT: __ __ __ __ f0 f1 __
}

int main() {
  // This has binary representation 0x00804020, which in memory (little-endian)
  // is {0x20,0x40,0x80,0x00}.
  float f = 1.17779472238e-38f;
  DoNotOptimize(f);
  char buffer[sizeof(float)];
  memcpy(buffer, &f, sizeof(float));
  printf("{0x%x, 0x%x, 0x%x, 0x%x}\n",
         (unsigned char)buffer[0], (unsigned char)buffer[1],
         (unsigned char)buffer[2], (unsigned char)buffer[3]);
#define str(s) #s
#define xstr(s) str(s)
  puts(xstr(FN));
  __nsan_dump_shadow_mem(buffer, sizeof(float), sizeof(float), 0);
  FN(buffer);
  return 0;
}
