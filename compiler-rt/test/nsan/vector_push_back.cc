// RUN: %clangxx_nsan -fno-builtin -O2 -g0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// This test verifies that dynamic memory is correctly tracked.

#include <cstddef>
#include <vector>

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes,
                                       size_t bytes_per_line, size_t reserved);

int main() {
  std::vector<double> values;
  values.push_back(1.028);
  __nsan_dump_shadow_mem((const char *)values.data(), 8, 8, 0);
  // CHECK: 0x{{[a-f0-9]*}}:    d0 d1 d2 d3 d4 d5 d6 d7 (1.02800000000000002487)
}
