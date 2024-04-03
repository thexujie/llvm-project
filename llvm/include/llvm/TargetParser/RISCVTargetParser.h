//===-- RISCVTargetParser - Parser for target features ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features
// for RISC-V CPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_RISCVTARGETPARSER_H
#define LLVM_TARGETPARSER_RISCVTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

// Sync with https://docs.kernel.org/arch/riscv/hwprobe.html
// and compiler-rt/lib/builtins/riscv/ifunc_select.c
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_BASE_BEHAVIOR_IMA (1 << 0)
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define RISCV_HWPROBE_IMA_FD (1 << 0)
#define RISCV_HWPROBE_IMA_C (1 << 1)
#define RISCV_HWPROBE_IMA_V (1 << 2)
#define RISCV_HWPROBE_EXT_ZBA (1 << 3)
#define RISCV_HWPROBE_EXT_ZBB (1 << 4)
#define RISCV_HWPROBE_EXT_ZBS (1 << 5)
#define RISCV_HWPROBE_EXT_ZICBOZ (1 << 6)
#define RISCV_HWPROBE_EXT_ZBC (1 << 7)
#define RISCV_HWPROBE_EXT_ZBKB (1 << 8)
#define RISCV_HWPROBE_EXT_ZBKC (1 << 9)
#define RISCV_HWPROBE_EXT_ZBKX (1 << 10)
#define RISCV_HWPROBE_EXT_ZKND (1 << 11)
#define RISCV_HWPROBE_EXT_ZKNE (1 << 12)
#define RISCV_HWPROBE_EXT_ZKNH (1 << 13)
#define RISCV_HWPROBE_EXT_ZKSED (1 << 14)
#define RISCV_HWPROBE_EXT_ZKSH (1 << 15)
#define RISCV_HWPROBE_EXT_ZKT (1 << 16)
#define RISCV_HWPROBE_EXT_ZVBB (1 << 17)
#define RISCV_HWPROBE_EXT_ZVBC (1 << 18)
#define RISCV_HWPROBE_EXT_ZVKB (1 << 19)
#define RISCV_HWPROBE_EXT_ZVKG (1 << 20)
#define RISCV_HWPROBE_EXT_ZVKNED (1 << 21)
#define RISCV_HWPROBE_EXT_ZVKNHA (1 << 22)
#define RISCV_HWPROBE_EXT_ZVKNHB (1 << 23)
#define RISCV_HWPROBE_EXT_ZVKSED (1 << 24)
#define RISCV_HWPROBE_EXT_ZVKSH (1 << 25)
#define RISCV_HWPROBE_EXT_ZVKT (1 << 26)
#define RISCV_HWPROBE_EXT_ZFH (1 << 27)
#define RISCV_HWPROBE_EXT_ZFHMIN (1 << 28)
#define RISCV_HWPROBE_EXT_ZIHINTNTL (1 << 29)
#define RISCV_HWPROBE_EXT_ZVFH (1 << 30)
#define RISCV_HWPROBE_EXT_ZVFHMIN (1 << 31)
#define RISCV_HWPROBE_EXT_ZFA (1ULL << 32)
#define RISCV_HWPROBE_EXT_ZTSO (1ULL << 33)
#define RISCV_HWPROBE_EXT_ZACAS (1ULL << 34)
#define RISCV_HWPROBE_EXT_ZICOND (1ULL << 35)

namespace llvm {

class Triple;

namespace RISCV {

// We use 64 bits as the known part in the scalable vector types.
static constexpr unsigned RVVBitsPerBlock = 64;

// RISC-V Hwprobe
const unsigned RISCVHwprobeLengthOfKey = 2;
const unsigned RISCVHwprobeKeyBase = RISCV_HWPROBE_KEY_BASE_BEHAVIOR;
const unsigned RISCVHwprobeKeyIMA = RISCV_HWPROBE_KEY_IMA_EXT_0;

void getFeaturesForCPU(StringRef CPU,
                       SmallVectorImpl<std::string> &EnabledFeatures,
                       bool NeedPlus = false);
bool parseCPU(StringRef CPU, bool IsRV64);
bool parseTuneCPU(StringRef CPU, bool IsRV64);
StringRef getMArchFromMcpu(StringRef CPU);
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64);
void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64);
bool hasFastUnalignedAccess(StringRef CPU);
std::vector<unsigned long long> getBaseExtensionKey(ArrayRef<StringRef>);
std::vector<unsigned long long>
    getIMACompatibleExtensionKey(ArrayRef<StringRef>);
llvm::SmallVector<std::string> getImpliedExts(StringRef Ext);

} // namespace RISCV

namespace RISCVII {
enum VLMUL : uint8_t {
  LMUL_1 = 0,
  LMUL_2,
  LMUL_4,
  LMUL_8,
  LMUL_RESERVED,
  LMUL_F8,
  LMUL_F4,
  LMUL_F2
};

enum {
  TAIL_UNDISTURBED_MASK_UNDISTURBED = 0,
  TAIL_AGNOSTIC = 1,
  MASK_AGNOSTIC = 2,
};
} // namespace RISCVII

namespace RISCVVType {
// Is this a SEW value that can be encoded into the VTYPE format.
inline static bool isValidSEW(unsigned SEW) {
  return isPowerOf2_32(SEW) && SEW >= 8 && SEW <= 1024;
}

// Is this a LMUL value that can be encoded into the VTYPE format.
inline static bool isValidLMUL(unsigned LMUL, bool Fractional) {
  return isPowerOf2_32(LMUL) && LMUL <= 8 && (!Fractional || LMUL != 1);
}

unsigned encodeVTYPE(RISCVII::VLMUL VLMUL, unsigned SEW, bool TailAgnostic,
                     bool MaskAgnostic);

inline static RISCVII::VLMUL getVLMUL(unsigned VType) {
  unsigned VLMUL = VType & 0x7;
  return static_cast<RISCVII::VLMUL>(VLMUL);
}

// Decode VLMUL into 1,2,4,8 and fractional indicator.
std::pair<unsigned, bool> decodeVLMUL(RISCVII::VLMUL VLMUL);

inline static RISCVII::VLMUL encodeLMUL(unsigned LMUL, bool Fractional) {
  assert(isValidLMUL(LMUL, Fractional) && "Unsupported LMUL");
  unsigned LmulLog2 = Log2_32(LMUL);
  return static_cast<RISCVII::VLMUL>(Fractional ? 8 - LmulLog2 : LmulLog2);
}

inline static unsigned decodeVSEW(unsigned VSEW) {
  assert(VSEW < 8 && "Unexpected VSEW value");
  return 1 << (VSEW + 3);
}

inline static unsigned encodeSEW(unsigned SEW) {
  assert(isValidSEW(SEW) && "Unexpected SEW value");
  return Log2_32(SEW) - 3;
}

inline static unsigned getSEW(unsigned VType) {
  unsigned VSEW = (VType >> 3) & 0x7;
  return decodeVSEW(VSEW);
}

inline static bool isTailAgnostic(unsigned VType) { return VType & 0x40; }

inline static bool isMaskAgnostic(unsigned VType) { return VType & 0x80; }

void printVType(unsigned VType, raw_ostream &OS);

unsigned getSEWLMULRatio(unsigned SEW, RISCVII::VLMUL VLMul);

std::optional<RISCVII::VLMUL>
getSameRatioLMUL(unsigned SEW, RISCVII::VLMUL VLMUL, unsigned EEW);
} // namespace RISCVVType

} // namespace llvm

#endif
