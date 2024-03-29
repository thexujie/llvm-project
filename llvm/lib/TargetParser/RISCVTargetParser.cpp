//===-- RISCVTargetParser.cpp - Parser for target features ------*- C++ -*-===//
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

#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace RISCV {

enum CPUKind : unsigned {
#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGN) CK_##ENUM,
#define TUNE_PROC(ENUM, NAME) CK_##ENUM,
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

struct CPUInfo {
  StringLiteral Name;
  StringLiteral DefaultMarch;
  bool FastUnalignedAccess;
  bool is64Bit() const { return DefaultMarch.starts_with("rv64"); }
};

constexpr CPUInfo RISCVCPUInfo[] = {
#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGN)                          \
  {NAME, DEFAULT_MARCH, FAST_UNALIGN},
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

static const CPUInfo *getCPUInfoByName(StringRef CPU) {
  for (auto &C : RISCVCPUInfo)
    if (C.Name == CPU)
      return &C;
  return nullptr;
}

bool hasFastUnalignedAccess(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  return Info && Info->FastUnalignedAccess;
}

bool parseCPU(StringRef CPU, bool IsRV64) {
  const CPUInfo *Info = getCPUInfoByName(CPU);

  if (!Info)
    return false;
  return Info->is64Bit() == IsRV64;
}

bool parseTuneCPU(StringRef TuneCPU, bool IsRV64) {
  std::optional<CPUKind> Kind =
      llvm::StringSwitch<std::optional<CPUKind>>(TuneCPU)
#define TUNE_PROC(ENUM, NAME) .Case(NAME, CK_##ENUM)
  #include "llvm/TargetParser/RISCVTargetParserDef.inc"
      .Default(std::nullopt);

  if (Kind.has_value())
    return true;

  // Fallback to parsing as a CPU.
  return parseCPU(TuneCPU, IsRV64);
}

StringRef getMArchFromMcpu(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  if (!Info)
    return "";
  return Info->DefaultMarch;
}

void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
}

void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
#define TUNE_PROC(ENUM, NAME) Values.emplace_back(StringRef(NAME));
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
}

// This function is currently used by IREE, so it's not dead code.
void getFeaturesForCPU(StringRef CPU,
                       SmallVectorImpl<std::string> &EnabledFeatures,
                       bool NeedPlus) {
  StringRef MarchFromCPU = llvm::RISCV::getMArchFromMcpu(CPU);
  if (MarchFromCPU == "")
    return;

  EnabledFeatures.clear();
  auto RII = RISCVISAInfo::parseArchString(
      MarchFromCPU, /* EnableExperimentalExtension */ true);

  if (llvm::errorToBool(RII.takeError()))
    return;

  std::vector<std::string> FeatStrings =
      (*RII)->toFeatures(/* AddAllExtensions */ false);
  for (const auto &F : FeatStrings)
    if (NeedPlus)
      EnabledFeatures.push_back(F);
    else
      EnabledFeatures.push_back(F.substr(1));
}

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

std::vector<unsigned long long> getBaseExtensionKey(ArrayRef<StringRef> Exts) {
  std::vector<unsigned long long> Result;
  for (auto Ext : Exts) {
    unsigned long long ExtKey = 0;
    if (Ext.starts_with("i") || Ext.starts_with("m") || Ext.starts_with("a")) {
      ExtKey = RISCV_HWPROBE_BASE_BEHAVIOR_IMA;
    }
    Result.push_back(ExtKey);
  }

  return Result;
}

std::vector<unsigned long long>
getIMACompatibleExtensionKey(ArrayRef<StringRef> Exts) {
  std::vector<unsigned long long> Result;

  for (auto Ext : Exts) {
    unsigned long long ExtKey = 0;
    if (Ext.starts_with("f") || Ext.starts_with("d"))
      ExtKey = RISCV_HWPROBE_IMA_FD;
    if (Ext.starts_with("c"))
      ExtKey = RISCV_HWPROBE_IMA_C;
    if (Ext.starts_with("v"))
      ExtKey = RISCV_HWPROBE_IMA_V;
    if (Ext.starts_with("zba"))
      ExtKey = RISCV_HWPROBE_EXT_ZBA;
    if (Ext.starts_with("zbb"))
      ExtKey = RISCV_HWPROBE_EXT_ZBB;
    if (Ext.starts_with("zbs"))
      ExtKey = RISCV_HWPROBE_EXT_ZBS;
    if (Ext.starts_with("zicboz"))
      ExtKey = RISCV_HWPROBE_EXT_ZICBOZ;
    if (Ext.starts_with("zbc"))
      ExtKey = RISCV_HWPROBE_EXT_ZBC;
    if (Ext.starts_with("zbkb"))
      ExtKey = RISCV_HWPROBE_EXT_ZBKB;
    if (Ext.starts_with("zbkc"))
      ExtKey = RISCV_HWPROBE_EXT_ZBKC;
    if (Ext.starts_with("zbkk"))
      ExtKey = RISCV_HWPROBE_EXT_ZBKX;
    if (Ext.starts_with("zknd"))
      ExtKey = RISCV_HWPROBE_EXT_ZKND;
    if (Ext.starts_with("zkne"))
      ExtKey = RISCV_HWPROBE_EXT_ZKNE;
    if (Ext.starts_with("zknh"))
      ExtKey = RISCV_HWPROBE_EXT_ZKNH;
    if (Ext.starts_with("zksed"))
      ExtKey = RISCV_HWPROBE_EXT_ZKSED;
    if (Ext.starts_with("zksh"))
      ExtKey = RISCV_HWPROBE_EXT_ZKSH;
    if (Ext.starts_with("zkt"))
      ExtKey = RISCV_HWPROBE_EXT_ZKT;
    if (Ext.starts_with("zvbb"))
      ExtKey = RISCV_HWPROBE_EXT_ZVBB;
    if (Ext.starts_with("zvbc"))
      ExtKey = RISCV_HWPROBE_EXT_ZVBC;
    if (Ext.starts_with("zvkb"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKB;
    if (Ext.starts_with("zvkg"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKG;
    if (Ext.starts_with("zvkned"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKNED;
    if (Ext.starts_with("zvknha"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKNHA;
    if (Ext.starts_with("zvknhb"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKNHB;
    if (Ext.starts_with("zvksed"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKSED;
    if (Ext.starts_with("zvksh"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKSH;
    if (Ext.starts_with("zvkt"))
      ExtKey = RISCV_HWPROBE_EXT_ZVKT;
    if (Ext.starts_with("zfh"))
      ExtKey = RISCV_HWPROBE_EXT_ZFH;
    if (Ext.starts_with("zfhmin"))
      ExtKey = RISCV_HWPROBE_EXT_ZFHMIN;
    if (Ext.starts_with("zihintntl"))
      ExtKey = RISCV_HWPROBE_EXT_ZIHINTNTL;
    if (Ext.starts_with("zvfh"))
      ExtKey = RISCV_HWPROBE_EXT_ZVFH;
    if (Ext.starts_with("zvfhmin"))
      ExtKey = RISCV_HWPROBE_EXT_ZVFHMIN;
    if (Ext.starts_with("zfa"))
      ExtKey = RISCV_HWPROBE_EXT_ZFA;
    if (Ext.starts_with("ztso"))
      ExtKey = RISCV_HWPROBE_EXT_ZTSO;
    if (Ext.starts_with("zacas"))
      ExtKey = RISCV_HWPROBE_EXT_ZACAS;
    if (Ext.starts_with("zicond"))
      ExtKey = RISCV_HWPROBE_EXT_ZICOND;

    Result.push_back(ExtKey);
  }

  return Result;
}

llvm::SmallVector<std::string> getImpliedExts(StringRef Ext) {
  return RISCVISAInfo::getImpliedExts(Ext);
}

} // namespace RISCV

namespace RISCVVType {
// Encode VTYPE into the binary format used by the the VSETVLI instruction which
// is used by our MC layer representation.
//
// Bits | Name       | Description
// -----+------------+------------------------------------------------
// 7    | vma        | Vector mask agnostic
// 6    | vta        | Vector tail agnostic
// 5:3  | vsew[2:0]  | Standard element width (SEW) setting
// 2:0  | vlmul[2:0] | Vector register group multiplier (LMUL) setting
unsigned encodeVTYPE(RISCVII::VLMUL VLMUL, unsigned SEW, bool TailAgnostic,
                     bool MaskAgnostic) {
  assert(isValidSEW(SEW) && "Invalid SEW");
  unsigned VLMULBits = static_cast<unsigned>(VLMUL);
  unsigned VSEWBits = encodeSEW(SEW);
  unsigned VTypeI = (VSEWBits << 3) | (VLMULBits & 0x7);
  if (TailAgnostic)
    VTypeI |= 0x40;
  if (MaskAgnostic)
    VTypeI |= 0x80;

  return VTypeI;
}

std::pair<unsigned, bool> decodeVLMUL(RISCVII::VLMUL VLMUL) {
  switch (VLMUL) {
  default:
    llvm_unreachable("Unexpected LMUL value!");
  case RISCVII::VLMUL::LMUL_1:
  case RISCVII::VLMUL::LMUL_2:
  case RISCVII::VLMUL::LMUL_4:
  case RISCVII::VLMUL::LMUL_8:
    return std::make_pair(1 << static_cast<unsigned>(VLMUL), false);
  case RISCVII::VLMUL::LMUL_F2:
  case RISCVII::VLMUL::LMUL_F4:
  case RISCVII::VLMUL::LMUL_F8:
    return std::make_pair(1 << (8 - static_cast<unsigned>(VLMUL)), true);
  }
}

void printVType(unsigned VType, raw_ostream &OS) {
  unsigned Sew = getSEW(VType);
  OS << "e" << Sew;

  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(getVLMUL(VType));

  if (Fractional)
    OS << ", mf";
  else
    OS << ", m";
  OS << LMul;

  if (isTailAgnostic(VType))
    OS << ", ta";
  else
    OS << ", tu";

  if (isMaskAgnostic(VType))
    OS << ", ma";
  else
    OS << ", mu";
}

unsigned getSEWLMULRatio(unsigned SEW, RISCVII::VLMUL VLMul) {
  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(VLMul);

  // Convert LMul to a fixed point value with 3 fractional bits.
  LMul = Fractional ? (8 / LMul) : (LMul * 8);

  assert(SEW >= 8 && "Unexpected SEW value");
  return (SEW * 8) / LMul;
}

std::optional<RISCVII::VLMUL>
getSameRatioLMUL(unsigned SEW, RISCVII::VLMUL VLMUL, unsigned EEW) {
  unsigned Ratio = RISCVVType::getSEWLMULRatio(SEW, VLMUL);
  unsigned EMULFixedPoint = (EEW * 8) / Ratio;
  bool Fractional = EMULFixedPoint < 8;
  unsigned EMUL = Fractional ? 8 / EMULFixedPoint : EMULFixedPoint / 8;
  if (!isValidLMUL(EMUL, Fractional))
    return std::nullopt;
  return RISCVVType::encodeLMUL(EMUL, Fractional);
}

} // namespace RISCVVType

} // namespace llvm
