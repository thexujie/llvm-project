// RUN: %clangxx_nsan -fno-builtin -mllvm -nsan-check-loads -O2 -g2 -UNDEBUG %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This verifies that shadow memory is tracked correcty across typed and
// bitcasted swaps.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes,
                                       size_t bytes_per_line, size_t reserved);

__attribute__((noinline))  // prevent optimization
void SwapFT(double *a, double* b) {
  // LLVM typically optimizes this to an untyped swap (through i64) anyway.
  std::swap(*a, *b);
}

__attribute__((noinline))  // prevent optimization
void SwapBitcasted(uint64_t *a, uint64_t*b) {
  std::swap(*a, *b);
}

int main() {
  double a = 1.0, b = 2.0;
  __nsan_dump_shadow_mem((const char *)&a, sizeof(a), sizeof(a), 0);
  __nsan_dump_shadow_mem((const char *)&b, sizeof(b), sizeof(b), 0);
  SwapFT(&a, &b);
  __nsan_dump_shadow_mem((const char *)&a, sizeof(a), sizeof(a), 0);
  __nsan_dump_shadow_mem((const char *)&b, sizeof(b), sizeof(b), 0);
  assert(a == 2.0 && b == 1.0);
  // This breaks strict aliasing but is OK on X86.
  SwapBitcasted(reinterpret_cast<uint64_t *>(&a),
                reinterpret_cast<uint64_t *>(&b));
  __nsan_dump_shadow_mem((const char *)&a, sizeof(a), sizeof(a), 0);
  __nsan_dump_shadow_mem((const char *)&b, sizeof(b), sizeof(b), 0);
  assert(a == 1.0 && b == 2.0);
  // CHECK: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7   (1.0{{.*}}
  // CHECK-NEXT: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7   (2.0{{.*}}
  // CHECK-NEXT: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7   (2.0{{.*}}
  // CHECK-NEXT: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7   (1.0{{.*}}
  // CHECK-NEXT: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7   (1.0{{.*}}
  // CHECK-NEXT: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7   (2.0{{.*}}
}
