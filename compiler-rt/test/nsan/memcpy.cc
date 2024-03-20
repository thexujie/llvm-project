// This test case verifies that we can track shadow memory values across
// explicit or implicit calls to memcpy.

// RUN: %clangxx_nsan -O2 -g -DIMPL=OpEq %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DIMPL=Memcpy %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_nsan -O2 -g -DIMPL=MemcpyInline %s -o %t && NSAN_OPTIONS=halt_on_error=0,resume_after_warning=false %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out


#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "helpers.h"

class OpEq {
 public:
  double* data() const { return data_.get();}

  OpEq() = default;
  OpEq(const OpEq& other) {
    *data_ = *other.data_;
  }

 private:
  std::unique_ptr<double> data_ = std::make_unique<double>();
};

class Memcpy {
 public:
  double* data() const { return data_.get();}

  Memcpy() = default;
  Memcpy(const Memcpy& other) {
    auto size = sizeof(double);
    DoNotOptimize(size);  // Prevent the compiler from optimizing this to a load-store.
    memcpy(data_.get(), other.data_.get(), size);
  }

 private:
  std::unique_ptr<double> data_ = std::make_unique<double>();
};

class MemcpyInline {
 public:
  double* data() const { return data_.get();}

  MemcpyInline() = default;
  MemcpyInline(const MemcpyInline& other) {
    __builtin_memcpy(data_.get(), other.data_.get(), sizeof(double));
  }

 private:
  std::unique_ptr<double> data_ = std::make_unique<double>();
};

class Vector : public std::vector<double> {
 public:
  Vector() : std::vector<double>(1) {}
};

int main() {
  using Impl = IMPL;
  Impl src;
  CreateInconsistency(src.data());
  DoNotOptimize(src);
  // We first verify that an incorrect value has been generated in the original
  // data location.
  printf("%.16f\n", *src.data());
  // CHECK: #0{{.*}}in main{{.*}}memcpy.cc:[[@LINE-1]]
  Impl dst(src);
  DoNotOptimize(dst);
  // This will fail if we correctly carried the shadow value across the copy.
  printf("%.16f\n", *dst.data());
  // CHECK: #0{{.*}}in main{{.*}}memcpy.cc:[[@LINE-1]]
  return 0;
}
