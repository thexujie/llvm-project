//===-- NumericalStabilitySanitizer.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of NumericalStabilitySanitizer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/NumericalStabilitySanitizer.h"

#include <cstdint>
#include <unordered_map>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "nsan"

STATISTIC(NumInstrumentedFTLoads,
          "Number of instrumented floating-point loads");

STATISTIC(NumInstrumentedFTCalls,
          "Number of instrumented floating-point calls");
STATISTIC(NumInstrumentedFTRets,
          "Number of instrumented floating-point returns");
STATISTIC(NumInstrumentedFTStores,
          "Number of instrumented floating-point stores");
STATISTIC(NumInstrumentedNonFTStores,
          "Number of instrumented non floating-point stores");
STATISTIC(
    NumInstrumentedNonFTMemcpyStores,
    "Number of instrumented non floating-point stores with memcpy semantics");
STATISTIC(NumInstrumentedFCmp, "Number of instrumented fcmps");

// Using smaller shadow types types can help improve speed. For example, `dlq`
// is 3x slower to 5x faster in opt mode and 2-6x faster in dbg mode compared to
// `dqq`.
static cl::opt<std::string> ClShadowMapping(
    "nsan-shadow-type-mapping", cl::init("dqq"),
    cl::desc("One shadow type id for each of `float`, `double`, `long double`. "
             "`d`,`l`,`q`,`e` mean double, x86_fp80, fp128 (quad) and "
             "ppc_fp128 (extended double) respectively. The default is to "
             "shadow `float` as `double`, and `double` and `x86_fp80` as "
             "`fp128`"),
    cl::Hidden);

static cl::opt<bool>
    ClInstrumentFCmp("nsan-instrument-fcmp", cl::init(true),
                     cl::desc("Instrument floating-point comparisons"),
                     cl::Hidden);

static cl::opt<std::string> ClCheckFunctionsFilter(
    "check-functions-filter",
    cl::desc("Only emit checks for arguments of functions "
             "whose names match the given regular expression"),
    cl::value_desc("regex"));

static cl::opt<bool> ClTruncateFCmpEq(
    "nsan-truncate-fcmp-eq", cl::init(true),
    cl::desc(
        "This flag controls the behaviour of fcmp equality comparisons:"
        "For equality comparisons such as `x == 0.0f`, we can perform the "
        "shadow check in the shadow (`x_shadow == 0.0) == (x == 0.0f)`) or app "
        " domain (`(trunc(x_shadow) == 0.0f) == (x == 0.0f)`). This helps "
        "catch the case when `x_shadow` is accurate enough (and therefore "
        "close enough to zero) so that `trunc(x_shadow)` is zero even though "
        "both `x` and `x_shadow` are not. "),
    cl::Hidden);

// When there is external, uninstrumented code writing to memory, the shadow
// memory can get out of sync with the application memory. Enabling this flag
// emits consistency checks for loads to catch this situation.
// When everything is instrumented, this is not strictly necessary because any
// load should have a corresponding store, but can help debug cases when the
// framework did a bad job at tracking shadow memory modifications by failing on
// load rather than store.
// FIXME: provide a way to resume computations from the FT value when the load
// is inconsistent. This ensures that further computations are not polluted.
static cl::opt<bool> ClCheckLoads("nsan-check-loads", cl::init(false),
                                  cl::desc("Check floating-point load"),
                                  cl::Hidden);

static cl::opt<bool> ClCheckStores("nsan-check-stores", cl::init(true),
                                   cl::desc("Check floating-point stores"),
                                   cl::Hidden);

static cl::opt<bool> ClCheckRet("nsan-check-ret", cl::init(true),
                                cl::desc("Check floating-point return values"),
                                cl::Hidden);

static const char *const kNsanModuleCtorName = "nsan.module_ctor";
static const char *const kNsanInitName = "__nsan_init";

// The following values must be kept in sync with the runtime.
static constexpr const int kShadowScale = 2;
static constexpr const int kMaxVectorWidth = 8;
static constexpr const int kMaxNumArgs = 128;
static constexpr const int kMaxShadowTypeSizeBytes = 16; // fp128

namespace {

// Defines the characteristics (type id, type, and floating-point semantics)
// attached for all possible shadow types.
class ShadowTypeConfig {
public:
  static std::unique_ptr<ShadowTypeConfig> fromNsanTypeId(char TypeId);
  // The floating-point semantics of the shadow type.
  virtual const fltSemantics &semantics() const = 0;

  // The LLVM Type corresponding to the shadow type.
  virtual Type *getType(LLVMContext &Context) const = 0;

  // The nsan type id of the shadow type (`d`, `l`, `q`, ...).
  virtual char getNsanTypeId() const = 0;

  virtual ~ShadowTypeConfig() {}
};

template <char NsanTypeId>
class ShadowTypeConfigImpl : public ShadowTypeConfig {
public:
  char getNsanTypeId() const override { return NsanTypeId; }
  static constexpr const char kNsanTypeId = NsanTypeId;
};

// `double` (`d`) shadow type.
class F64ShadowConfig : public ShadowTypeConfigImpl<'d'> {
  const fltSemantics &semantics() const override {
    return APFloat::IEEEdouble();
  }
  Type *getType(LLVMContext &Context) const override {
    return Type::getDoubleTy(Context);
  }
};

// `x86_fp80` (`l`) shadow type: X86 long double.
class F80ShadowConfig : public ShadowTypeConfigImpl<'l'> {
  const fltSemantics &semantics() const override {
    return APFloat::x87DoubleExtended();
  }
  Type *getType(LLVMContext &Context) const override {
    return Type::getX86_FP80Ty(Context);
  }
};

// `fp128` (`q`) shadow type.
class F128ShadowConfig : public ShadowTypeConfigImpl<'q'> {
  const fltSemantics &semantics() const override { return APFloat::IEEEquad(); }
  Type *getType(LLVMContext &Context) const override {
    return Type::getFP128Ty(Context);
  }
};

// `ppc_fp128` (`e`) shadow type: IBM extended double with 106 bits of mantissa.
class PPC128ShadowConfig : public ShadowTypeConfigImpl<'e'> {
  const fltSemantics &semantics() const override {
    return APFloat::PPCDoubleDouble();
  }
  Type *getType(LLVMContext &Context) const override {
    return Type::getPPC_FP128Ty(Context);
  }
};

// Creates a ShadowTypeConfig given its type id.
std::unique_ptr<ShadowTypeConfig>
ShadowTypeConfig::fromNsanTypeId(const char TypeId) {
  switch (TypeId) {
  case F64ShadowConfig::kNsanTypeId:
    return std::make_unique<F64ShadowConfig>();
  case F80ShadowConfig::kNsanTypeId:
    return std::make_unique<F80ShadowConfig>();
  case F128ShadowConfig::kNsanTypeId:
    return std::make_unique<F128ShadowConfig>();
  case PPC128ShadowConfig::kNsanTypeId:
    return std::make_unique<PPC128ShadowConfig>();
  }
  errs() << "nsan: invalid shadow type id'" << TypeId << "'\n";
  return nullptr;
}

// An enum corresponding to shadow value types. Used as indices in arrays, so
// not an `enum class`.
enum FTValueType { kFloat, kDouble, kLongDouble, kNumValueTypes };

static FTValueType semanticsToFTValueType(const fltSemantics &Sem) {
  if (&Sem == &APFloat::IEEEsingle()) {
    return kFloat;
  } else if (&Sem == &APFloat::IEEEdouble()) {
    return kDouble;
  } else if (&Sem == &APFloat::x87DoubleExtended()) {
    return kLongDouble;
  }
  llvm_unreachable("semantics are not one of the handled types");
}

// If `FT` corresponds to a primitive FTValueType, return it.
static std::optional<FTValueType> ftValueTypeFromType(Type *FT) {
  if (FT->isFloatTy())
    return kFloat;
  if (FT->isDoubleTy())
    return kDouble;
  if (FT->isX86_FP80Ty())
    return kLongDouble;
  return {};
}

// Returns the LLVM type for an FTValueType.
static Type *typeFromFTValueType(FTValueType VT, LLVMContext &Context) {
  switch (VT) {
  case kFloat:
    return Type::getFloatTy(Context);
  case kDouble:
    return Type::getDoubleTy(Context);
  case kLongDouble:
    return Type::getX86_FP80Ty(Context);
  case kNumValueTypes:
    return nullptr;
  }
}

// Returns the type name for an FTValueType.
static const char *typeNameFromFTValueType(FTValueType VT) {
  switch (VT) {
  case kFloat:
    return "float";
  case kDouble:
    return "double";
  case kLongDouble:
    return "longdouble";
  case kNumValueTypes:
    return nullptr;
  }
}

// A specific mapping configuration of application type to shadow type for nsan
// (see -nsan-shadow-mapping flag).
class MappingConfig {
public:
  bool initialize(LLVMContext *C) {
    if (ClShadowMapping.size() != 3) {
      errs() << "Invalid nsan mapping: " << ClShadowMapping << "\n";
    }
    Context = C;
    unsigned ShadowTypeSizeBits[kNumValueTypes];
    for (int VT = 0; VT < kNumValueTypes; ++VT) {
      auto Config = ShadowTypeConfig::fromNsanTypeId(ClShadowMapping[VT]);
      if (Config == nullptr)
        return false;
      const unsigned AppTypeSize =
          typeFromFTValueType(static_cast<FTValueType>(VT), *C)
              ->getScalarSizeInBits();
      const unsigned ShadowTypeSize =
          Config->getType(*C)->getScalarSizeInBits();
      // Check that the shadow type size is at most kShadowScale times the
      // application type size, so that shadow memory compoutations are valid.
      if (ShadowTypeSize > kShadowScale * AppTypeSize) {
        errs() << "Invalid nsan mapping f" << AppTypeSize << "->f"
               << ShadowTypeSize << ": The shadow type size should be at most "
               << kShadowScale << " times the application type size\n";
        return false;
      }
      ShadowTypeSizeBits[VT] = ShadowTypeSize;
      Configs[VT] = std::move(Config);
    }

    // Check that the mapping is monotonous. This is required because if one
    // does an fpextend of `float->long double` in application code, nsan is
    // going to do an fpextend of `shadow(float) -> shadow(long double)` in
    // shadow code. This will fail in `qql` mode, since nsan would be
    // fpextending `f128->long`, which is invalid.
    // FIXME: Relax this.
    if (ShadowTypeSizeBits[kFloat] > ShadowTypeSizeBits[kDouble] ||
        ShadowTypeSizeBits[kDouble] > ShadowTypeSizeBits[kLongDouble]) {
      errs() << "Invalid nsan mapping: { float->f" << ShadowTypeSizeBits[kFloat]
             << "; double->f" << ShadowTypeSizeBits[kDouble]
             << "; long double->f" << ShadowTypeSizeBits[kLongDouble] << " }\n";
      return false;
    }
    return true;
  }

  const ShadowTypeConfig &byValueType(FTValueType VT) const {
    assert(VT < FTValueType::kNumValueTypes && "invalid value type");
    return *Configs[VT];
  }

  const ShadowTypeConfig &bySemantics(const fltSemantics &Sem) const {
    return byValueType(semanticsToFTValueType(Sem));
  }

  // Returns the extended shadow type for a given application type.
  Type *getExtendedFPType(Type *FT) const {
    if (const auto VT = ftValueTypeFromType(FT))
      return Configs[*VT]->getType(*Context);
    if (FT->isVectorTy()) {
      auto *VecTy = cast<VectorType>(FT);
      Type *ExtendedScalar = getExtendedFPType(VecTy->getElementType());
      return ExtendedScalar
                 ? VectorType::get(ExtendedScalar, VecTy->getElementCount())
                 : nullptr;
    }
    return nullptr;
  }

private:
  LLVMContext *Context = nullptr;
  std::unique_ptr<ShadowTypeConfig> Configs[FTValueType::kNumValueTypes];
};

// The memory extents of a type specifies how many elements of a given
// FTValueType needs to be stored when storing this type.
struct MemoryExtents {
  FTValueType ValueType;
  uint64_t NumElts;
};
static MemoryExtents getMemoryExtentsOrDie(Type *FT) {
  if (const auto VT = ftValueTypeFromType(FT))
    return {*VT, 1};
  if (FT->isVectorTy()) {
    auto *VecTy = cast<VectorType>(FT);
    const auto ScalarExtents = getMemoryExtentsOrDie(VecTy->getElementType());
    return {ScalarExtents.ValueType,
            ScalarExtents.NumElts * VecTy->getElementCount().getFixedValue()};
  }
  llvm_unreachable("invalid value type");
}

// The location of a check. Passed as parameters to runtime checking functions.
class CheckLoc {
public:
  // Creates a location that references an application memory location.
  static CheckLoc makeStore(Value *Address) {
    CheckLoc Result(kStore);
    Result.Address = Address;
    return Result;
  }
  static CheckLoc makeLoad(Value *Address) {
    CheckLoc Result(kLoad);
    Result.Address = Address;
    return Result;
  }

  // Creates a location that references an argument, given by id.
  static CheckLoc makeArg(int ArgId) {
    CheckLoc Result(kArg);
    Result.ArgId = ArgId;
    return Result;
  }

  // Creates a location that references the return value of a function.
  static CheckLoc makeRet() { return CheckLoc(kRet); }

  // Creates a location that references a vector insert.
  static CheckLoc makeInsert() { return CheckLoc(kInsert); }

  // Returns the CheckType of location this refers to, as an integer-typed LLVM
  // IR value.
  Value *getType(LLVMContext &C) const {
    return ConstantInt::get(Type::getInt32Ty(C), static_cast<int>(CheckTy));
  }

  // Returns a CheckType-specific value representing details of the location
  // (e.g. application address for loads or stores), as an `IntptrTy`-typed LLVM
  // IR value.
  Value *getValue(Type *IntptrTy, IRBuilder<> &Builder) const {
    switch (CheckTy) {
    case kUnknown:
      llvm_unreachable("unknown type");
    case kRet:
    case kInsert:
      return ConstantInt::get(IntptrTy, 0);
    case kArg:
      return ConstantInt::get(IntptrTy, ArgId);
    case kLoad:
    case kStore:
      return Builder.CreatePtrToInt(Address, IntptrTy);
    }
  }

private:
  // Must be kept in sync with the runtime.
  enum CheckType {
    kUnknown = 0,
    kRet,
    kArg,
    kLoad,
    kStore,
    kInsert,
  };
  explicit CheckLoc(CheckType CheckTy) : CheckTy(CheckTy) {}

  const CheckType CheckTy;
  Value *Address = nullptr;
  int ArgId = -1;
};

// A map of LLVM IR values to shadow LLVM IR values.
class ValueToShadowMap {
public:
  explicit ValueToShadowMap(MappingConfig *Config) : Config(Config) {}

  // Sets the shadow value for a value. Asserts that the value does not already
  // have a value.
  void setShadow(Value *V, Value *Shadow) {
    assert(V);
    assert(Shadow);
    const bool Inserted = Map.emplace(V, Shadow).second;
#ifdef LLVM_ENABLE_DUMP
    if (!Inserted) {
      if (const auto *const I = dyn_cast<Instruction>(V))
        I->getParent()->getParent()->dump();
      errs() << "duplicate shadow (" << V << "): ";
      V->dump();
    }
#endif
    assert(Inserted && "duplicate shadow");
    (void)Inserted;
  }

  // Returns true if the value already has a shadow (including if the value is a
  // constant). If true, calling getShadow() is valid.
  bool hasShadow(Value *V) const {
    return isa<Constant>(V) || (Map.find(V) != Map.end());
  }

  // Returns the shadow value for a given value. Asserts that the value has
  // a shadow value. Lazily creates shadows for constant values.
  Value *getShadow(Value *V) const {
    assert(V);
    if (Constant *C = dyn_cast<Constant>(V))
      return getShadowConstant(C);
    const auto ShadowValIt = Map.find(V);
    assert(ShadowValIt != Map.end() && "shadow val does not exist");
    assert(ShadowValIt->second && "shadow val is null");
    return ShadowValIt->second;
  }

  bool empty() const { return Map.empty(); }

private:
  // Extends a constant application value to its shadow counterpart.
  APFloat extendConstantFP(APFloat CV) const {
    bool LosesInfo = false;
    CV.convert(Config->bySemantics(CV.getSemantics()).semantics(),
               APFloatBase::rmTowardZero, &LosesInfo);
    return CV;
  }

  // Returns the shadow constant for the given application constant.
  Constant *getShadowConstant(Constant *C) const {
    if (UndefValue *U = dyn_cast<UndefValue>(C)) {
      return UndefValue::get(Config->getExtendedFPType(U->getType()));
    }
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      // Floating-point constants.
      return ConstantFP::get(Config->getExtendedFPType(CFP->getType()),
                             extendConstantFP(CFP->getValueAPF()));
    }
    // Vector, array, or aggregate constants.
    if (C->getType()->isVectorTy()) {
      SmallVector<Constant *, 8> Elements;
      for (int I = 0, E = cast<VectorType>(C->getType())
                              ->getElementCount()
                              .getFixedValue();
           I < E; ++I)
        Elements.push_back(getShadowConstant(C->getAggregateElement(I)));
      return ConstantVector::get(Elements);
    }
    llvm_unreachable("unimplemented");
  }

  MappingConfig *const Config;
  std::unordered_map<Value *, Value *> Map;
};

/// Instantiating NumericalStabilitySanitizer inserts the nsan runtime library
/// API function declarations into the module if they don't exist already.
/// Instantiating ensures the __nsan_init function is in the list of global
/// constructors for the module.
class NumericalStabilitySanitizer {
public:
  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);

private:
  void initialize(Module &M);
  bool instrumentMemIntrinsic(MemIntrinsic *MI);
  void maybeAddSuffixForNsanInterface(CallBase *CI);
  bool addrPointsToConstantData(Value *Addr);
  void maybeCreateShadowValue(Instruction &Root, const TargetLibraryInfo &TLI,
                              ValueToShadowMap &Map);
  Value *createShadowValueWithOperandsAvailable(Instruction &Inst,
                                                const TargetLibraryInfo &TLI,
                                                const ValueToShadowMap &Map);
  PHINode *maybeCreateShadowPhi(PHINode &Phi, const TargetLibraryInfo &TLI);
  void createShadowArguments(Function &F, const TargetLibraryInfo &TLI,
                             ValueToShadowMap &Map);

  void populateShadowStack(CallBase &CI, const TargetLibraryInfo &TLI,
                           const ValueToShadowMap &Map);

  void propagateShadowValues(Instruction &Inst, const TargetLibraryInfo &TLI,
                             const ValueToShadowMap &Map);
  Value *emitCheck(Value *V, Value *ShadowV, IRBuilder<> &Builder,
                   CheckLoc Loc);
  Value *emitCheckInternal(Value *V, Value *ShadowV, IRBuilder<> &Builder,
                           CheckLoc Loc);
  void emitFCmpCheck(FCmpInst &FCmp, const ValueToShadowMap &Map);
  Value *getCalleeAddress(CallBase &Call, IRBuilder<> &Builder) const;

  // Value creation handlers.
  Value *handleLoad(LoadInst &Load, Type *VT, Type *ExtendedVT);
  Value *handleTrunc(FPTruncInst &Trunc, Type *VT, Type *ExtendedVT,
                     const ValueToShadowMap &Map);
  Value *handleExt(FPExtInst &Ext, Type *VT, Type *ExtendedVT,
                   const ValueToShadowMap &Map);
  Value *handleCallBase(CallBase &Call, Type *VT, Type *ExtendedVT,
                        const TargetLibraryInfo &TLI,
                        const ValueToShadowMap &Map, IRBuilder<> &Builder);
  Value *maybeHandleKnownCallBase(CallBase &Call, Type *VT, Type *ExtendedVT,
                                  const TargetLibraryInfo &TLI,
                                  const ValueToShadowMap &Map,
                                  IRBuilder<> &Builder);

  // Value propagation handlers.
  void propagateFTStore(StoreInst &Store, Type *VT, Type *ExtendedVT,
                        const ValueToShadowMap &Map);
  void propagateNonFTStore(StoreInst &Store, Type *VT,
                           const ValueToShadowMap &Map);

  MappingConfig Config;
  LLVMContext *Context = nullptr;
  IntegerType *IntptrTy = nullptr;
  FunctionCallee NsanGetShadowPtrForStore[FTValueType::kNumValueTypes];
  FunctionCallee NsanGetShadowPtrForLoad[FTValueType::kNumValueTypes];
  FunctionCallee NsanCheckValue[FTValueType::kNumValueTypes];
  FunctionCallee NsanFCmpFail[FTValueType::kNumValueTypes];
  FunctionCallee NsanCopyValues;
  FunctionCallee NsanSetValueUnknown;
  FunctionCallee NsanGetRawShadowTypePtr;
  FunctionCallee NsanGetRawShadowPtr;
  GlobalValue *NsanShadowRetTag;

  Type *NsanShadowRetType;
  GlobalValue *NsanShadowRetPtr;

  GlobalValue *NsanShadowArgsTag;

  Type *NsanShadowArgsType;
  GlobalValue *NsanShadowArgsPtr;

  std::optional<Regex> CheckFunctionsFilter;
};

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kNsanModuleCtorName, kNsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}

} // end anonymous namespace

PreservedAnalyses
NumericalStabilitySanitizerPass::run(Function &F,
                                     FunctionAnalysisManager &FAM) {
  NumericalStabilitySanitizer Nsan;
  if (Nsan.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses
NumericalStabilitySanitizerPass::run(Module &M, ModuleAnalysisManager &MAM) {
  insertModuleCtor(M);
  return PreservedAnalyses::none();
}

static GlobalValue *createThreadLocalGV(const char *Name, Module &M, Type *Ty) {
  return dyn_cast<GlobalValue>(M.getOrInsertGlobal(Name, Ty, [&M, Ty, Name] {
    return new GlobalVariable(M, Ty, false, GlobalVariable::ExternalLinkage,
                              nullptr, Name, nullptr,
                              GlobalVariable::InitialExecTLSModel);
  }));
}

void NumericalStabilitySanitizer::initialize(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  Context = &M.getContext();
  IntptrTy = DL.getIntPtrType(*Context);
  Type *PtrTy = PointerType::getUnqual(*Context);
  Type *Int32Ty = Type::getInt32Ty(*Context);
  Type *Int1Ty = Type::getInt1Ty(*Context);
  Type *VoidTy = Type::getVoidTy(*Context);

  AttributeList Attr;
  Attr = Attr.addFnAttribute(*Context, Attribute::NoUnwind);
  // Initialize the runtime values (functions and global variables).
  for (int I = 0; I < kNumValueTypes; ++I) {
    const FTValueType VT = static_cast<FTValueType>(I);
    const char *const VTName = typeNameFromFTValueType(VT);
    Type *const VTTy = typeFromFTValueType(VT, *Context);

    // Load/store.
    const std::string GetterPrefix =
        std::string("__nsan_get_shadow_ptr_for_") + VTName;
    NsanGetShadowPtrForStore[VT] = M.getOrInsertFunction(
        GetterPrefix + "_store", Attr, PtrTy, PtrTy, IntptrTy);
    NsanGetShadowPtrForLoad[VT] = M.getOrInsertFunction(
        GetterPrefix + "_load", Attr, PtrTy, PtrTy, IntptrTy);

    // Check.
    const auto &ShadowConfig = Config.byValueType(VT);
    Type *ShadowTy = ShadowConfig.getType(*Context);
    NsanCheckValue[VT] =
        M.getOrInsertFunction(std::string("__nsan_internal_check_") + VTName +
                                  "_" + ShadowConfig.getNsanTypeId(),
                              Attr, Int32Ty, VTTy, ShadowTy, Int32Ty, IntptrTy);
    NsanFCmpFail[VT] = M.getOrInsertFunction(
        std::string("__nsan_fcmp_fail_") + VTName + "_" +
            ShadowConfig.getNsanTypeId(),
        Attr, VoidTy, VTTy, VTTy, ShadowTy, ShadowTy, Int32Ty, Int1Ty, Int1Ty);
  }

  NsanCopyValues = M.getOrInsertFunction("__nsan_copy_values", Attr, VoidTy,
                                         PtrTy, PtrTy, IntptrTy);
  NsanSetValueUnknown = M.getOrInsertFunction("__nsan_set_value_unknown", Attr,
                                              VoidTy, PtrTy, IntptrTy);

  // FIXME: Add attributes nofree, nosync, readnone, readonly,
  NsanGetRawShadowTypePtr = M.getOrInsertFunction(
      "__nsan_internal_get_raw_shadow_type_ptr", Attr, PtrTy, PtrTy);
  NsanGetRawShadowPtr = M.getOrInsertFunction(
      "__nsan_internal_get_raw_shadow_ptr", Attr, PtrTy, PtrTy);

  NsanShadowRetTag = createThreadLocalGV("__nsan_shadow_ret_tag", M, IntptrTy);

  NsanShadowRetType = ArrayType::get(Type::getInt8Ty(*Context),
                                     kMaxVectorWidth * kMaxShadowTypeSizeBytes);
  NsanShadowRetPtr =
      createThreadLocalGV("__nsan_shadow_ret_ptr", M, NsanShadowRetType);

  NsanShadowArgsTag =
      createThreadLocalGV("__nsan_shadow_args_tag", M, IntptrTy);

  NsanShadowArgsType =
      ArrayType::get(Type::getInt8Ty(*Context),
                     kMaxVectorWidth * kMaxNumArgs * kMaxShadowTypeSizeBytes);

  NsanShadowArgsPtr =
      createThreadLocalGV("__nsan_shadow_args_ptr", M, NsanShadowArgsType);

  if (!ClCheckFunctionsFilter.empty()) {
    Regex R = Regex(ClCheckFunctionsFilter);
    std::string RegexError;
    assert(R.isValid(RegexError));
    CheckFunctionsFilter = std::move(R);
  }
}

// Returns true if the given LLVM Value points to constant data (typically, a
// global variable reference).
bool NumericalStabilitySanitizer::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    return GV->isConstant();
  }
  return false;
}

// This instruments the function entry to create shadow arguments.
// Pseudocode:
//   if (this_fn_ptr == __nsan_shadow_args_tag) {
//     s(arg0) = LOAD<sizeof(arg0)>(__nsan_shadow_args);
//     s(arg1) = LOAD<sizeof(arg1)>(__nsan_shadow_args + sizeof(arg0));
//     ...
//     __nsan_shadow_args_tag = 0;
//   } else {
//     s(arg0) = fext(arg0);
//     s(arg1) = fext(arg1);
//     ...
//   }
void NumericalStabilitySanitizer::createShadowArguments(
    Function &F, const TargetLibraryInfo &TLI, ValueToShadowMap &Map) {
  assert(!F.getIntrinsicID() && "found a definition of an intrinsic");

  // Do not bother if there are no FP args.
  if (all_of(F.args(), [this](const Argument &Arg) {
        return Config.getExtendedFPType(Arg.getType()) == nullptr;
      }))
    return;

  const DataLayout &DL = F.getParent()->getDataLayout();
  IRBuilder<> Builder(F.getEntryBlock().getFirstNonPHI());
  // The function has shadow args if the shadow args tag matches the function
  // address.
  Value *HasShadowArgs = Builder.CreateICmpEQ(
      Builder.CreateLoad(IntptrTy, NsanShadowArgsTag, /*isVolatile=*/false),
      Builder.CreatePtrToInt(&F, IntptrTy));

  unsigned ShadowArgsOffsetBytes = 0;
  for (Argument &Arg : F.args()) {
    Type *const VT = Arg.getType();
    Type *const ExtendedVT = Config.getExtendedFPType(VT);
    if (ExtendedVT == nullptr)
      continue; // Not an FT value.
    Value *Shadow = Builder.CreateSelect(
        HasShadowArgs,
        Builder.CreateAlignedLoad(
            ExtendedVT,
            Builder.CreateConstGEP2_64(NsanShadowArgsType, NsanShadowArgsPtr, 0,
                                       ShadowArgsOffsetBytes),
            Align(1), /*isVolatile=*/false),
        Builder.CreateCast(Instruction::FPExt, &Arg, ExtendedVT));
    Map.setShadow(&Arg, Shadow);
    TypeSize SlotSize = DL.getTypeStoreSize(ExtendedVT);
    assert(!SlotSize.isScalable() && "unsupported");
    ShadowArgsOffsetBytes += SlotSize.getFixedValue();
  }
  Builder.CreateStore(ConstantInt::get(IntptrTy, 0), NsanShadowArgsTag);
}

// Returns true if the instrumentation should emit code to check arguments
// before a function call.
static bool shouldCheckArgs(CallBase &CI, const TargetLibraryInfo &TLI,
                            const std::optional<Regex> &CheckFunctionsFilter) {

  Function *Fn = CI.getCalledFunction();

  if (CheckFunctionsFilter) {
    // Skip checking args of indirect calls.
    if (Fn == nullptr)
      return false;
    if (CheckFunctionsFilter->match(Fn->getName()))
      return true;
    return false;
  }

  if (Fn == nullptr)
    return true; // Always check args of indirect calls.

  // Never check nsan functions, the user called them for a reason.
  if (Fn->getName().starts_with("__nsan_"))
    return false;

  const auto ID = Fn->getIntrinsicID();
  LibFunc LFunc = LibFunc::NumLibFuncs;
  // Always check args of unknown functions.
  if (ID == Intrinsic::ID() && !TLI.getLibFunc(*Fn, LFunc))
    return true;

  // Do not check args of an `fabs` call that is used for a comparison.
  // This is typically used for `fabs(a-b) < tolerance`, where what matters is
  // the result of the comparison, which is already caught be the fcmp checks.
  if (ID == Intrinsic::fabs || LFunc == LibFunc_fabsf ||
      LFunc == LibFunc_fabs || LFunc == LibFunc_fabsl)
    for (const auto &U : CI.users())
      if (isa<CmpInst>(U))
        return false;

  return true; // Default is check.
}

// Populates the shadow call stack (which contains shadow values for every
// floating-point parameter to the function).
void NumericalStabilitySanitizer::populateShadowStack(
    CallBase &CI, const TargetLibraryInfo &TLI, const ValueToShadowMap &Map) {
  // Do not create a shadow stack for inline asm.
  if (CI.isInlineAsm())
    return;

  // Do not bother if there are no FP args.
  if (all_of(CI.operands(), [this](const Value *Arg) {
        return Config.getExtendedFPType(Arg->getType()) == nullptr;
      }))
    return;

  IRBuilder<> Builder(&CI);
  SmallVector<Value *, 8> ArgShadows;
  const bool ShouldCheckArgs = shouldCheckArgs(CI, TLI, CheckFunctionsFilter);
  int ArgId = -1;
  for (Value *Arg : CI.operands()) {
    ++ArgId;
    if (Config.getExtendedFPType(Arg->getType()) == nullptr)
      continue; // Not an FT value.
    Value *ArgShadow = Map.getShadow(Arg);
    ArgShadows.push_back(ShouldCheckArgs ? emitCheck(Arg, ArgShadow, Builder,
                                                     CheckLoc::makeArg(ArgId))
                                         : ArgShadow);
  }

  // Do not create shadow stacks for intrinsics/known lib funcs.
  if (Function *Fn = CI.getCalledFunction()) {
    LibFunc LFunc;
    if (Fn->getIntrinsicID() || TLI.getLibFunc(*Fn, LFunc))
      return;
  }

  const DataLayout &DL =
      CI.getParent()->getParent()->getParent()->getDataLayout();
  // Set the shadow stack tag.
  Builder.CreateStore(getCalleeAddress(CI, Builder), NsanShadowArgsTag);
  unsigned ShadowArgsOffsetBytes = 0;

  unsigned ShadowArgId = 0;
  for (const Value *Arg : CI.operands()) {
    Type *const VT = Arg->getType();
    Type *const ExtendedVT = Config.getExtendedFPType(VT);
    if (ExtendedVT == nullptr)
      continue; // Not an FT value.
    Builder.CreateAlignedStore(
        ArgShadows[ShadowArgId++],
        Builder.CreateConstGEP2_64(NsanShadowArgsType, NsanShadowArgsPtr, 0,
                                   ShadowArgsOffsetBytes),
        Align(1), /*isVolatile=*/false);
    TypeSize SlotSize = DL.getTypeStoreSize(ExtendedVT);
    assert(!SlotSize.isScalable() && "unsupported");
    ShadowArgsOffsetBytes += SlotSize.getFixedValue();
  }
}

// Internal part of emitCheck(). Returns a value that indicates whether
// computation should continue with the shadow or resume by re-fextending the
// value.
enum ContinuationType { // Keep in sync with runtime.
  kContinueWithShadow = 0,
  kResumeFromValue = 1,
};
Value *NumericalStabilitySanitizer::emitCheckInternal(Value *V, Value *ShadowV,
                                                      IRBuilder<> &Builder,
                                                      CheckLoc Loc) {
  // Do not emit checks for constant values, this is redundant.
  if (isa<Constant>(V))
    return ConstantInt::get(Builder.getInt32Ty(), kContinueWithShadow);

  Type *const Ty = V->getType();
  if (const auto VT = ftValueTypeFromType(Ty))
    return Builder.CreateCall(
        NsanCheckValue[*VT],
        {V, ShadowV, Loc.getType(*Context), Loc.getValue(IntptrTy, Builder)});

  if (Ty->isVectorTy()) {
    auto *VecTy = cast<VectorType>(Ty);
    Value *CheckResult = nullptr;
    for (int I = 0, E = VecTy->getElementCount().getFixedValue(); I < E; ++I) {
      // We resume if any element resumes. Another option would be to create a
      // vector shuffle with the array of ContinueWithShadow, but that is too
      // complex.
      Value *ComponentCheckResult = emitCheckInternal(
          Builder.CreateExtractElement(V, I),
          Builder.CreateExtractElement(ShadowV, I), Builder, Loc);
      CheckResult = CheckResult
                        ? Builder.CreateOr(CheckResult, ComponentCheckResult)
                        : ComponentCheckResult;
    }
    return CheckResult;
  }
  if (Ty->isArrayTy()) {
    Value *CheckResult = nullptr;
    for (int I = 0, E = Ty->getArrayNumElements(); I < E; ++I) {
      Value *ComponentCheckResult = emitCheckInternal(
          Builder.CreateExtractValue(V, I),
          Builder.CreateExtractValue(ShadowV, I), Builder, Loc);
      CheckResult = CheckResult
                        ? Builder.CreateOr(CheckResult, ComponentCheckResult)
                        : ComponentCheckResult;
    }
    return CheckResult;
  }
  if (Ty->isStructTy()) {
    Value *CheckResult = nullptr;
    for (int I = 0, E = Ty->getStructNumElements(); I < E; ++I) {
      if (Config.getExtendedFPType(Ty->getStructElementType(I)) == nullptr)
        continue; // Only check FT values.
      Value *ComponentCheckResult = emitCheckInternal(
          Builder.CreateExtractValue(V, I),
          Builder.CreateExtractValue(ShadowV, I), Builder, Loc);
      CheckResult = CheckResult
                        ? Builder.CreateOr(CheckResult, ComponentCheckResult)
                        : ComponentCheckResult;
    }
    assert(CheckResult && "struct with no FT element");
    return CheckResult;
  }

  llvm_unreachable("not implemented");
}

// Inserts a runtime check of V against its shadow value ShadowV.
// We check values whenever they escape: on return, call, stores, and
// insertvalue.
// Returns the shadow value that should be used to continue the computations,
// depending on the answer from the runtime.
// FIXME: Should we check on select ? phi ?
Value *NumericalStabilitySanitizer::emitCheck(Value *V, Value *ShadowV,
                                              IRBuilder<> &Builder,
                                              CheckLoc Loc) {
  // Do not emit checks for constant values, this is redundant.
  if (isa<Constant>(V))
    return ShadowV;

  if (Instruction *Inst = dyn_cast<Instruction>(V)) {
    Function *F = Inst->getFunction();
    if (CheckFunctionsFilter &&
        !(F && CheckFunctionsFilter->match(F->getName()))) {
      return ShadowV;
    }
  }

  Value *CheckResult = emitCheckInternal(V, ShadowV, Builder, Loc);
  return Builder.CreateSelect(
      Builder.CreateICmpEQ(CheckResult, ConstantInt::get(Builder.getInt32Ty(),
                                                         kResumeFromValue)),
      Builder.CreateCast(Instruction::FPExt, V,
                         Config.getExtendedFPType(V->getType())),
      ShadowV);
}

static Instruction *getNextInstructionOrDie(Instruction &Inst) {
  assert(Inst.getNextNode() && "instruction is a terminator");
  return Inst.getNextNode();
}

// Inserts a check that fcmp on shadow values are consistent with that on base
// values.
void NumericalStabilitySanitizer::emitFCmpCheck(FCmpInst &FCmp,
                                                const ValueToShadowMap &Map) {
  if (!ClInstrumentFCmp)
    return;

  Function *F = FCmp.getFunction();
  if (CheckFunctionsFilter &&
      !(F && CheckFunctionsFilter->match(F->getName()))) {
    return;
  }

  Value *LHS = FCmp.getOperand(0);
  if (Config.getExtendedFPType(LHS->getType()) == nullptr)
    return;
  Value *RHS = FCmp.getOperand(1);

  // Split the basic block. On mismatch, we'll jump to the new basic block with
  // a call to the runtime for error reporting.
  BasicBlock *FCmpBB = FCmp.getParent();
  BasicBlock *NextBB = FCmpBB->splitBasicBlock(getNextInstructionOrDie(FCmp));
  // Remove the newly created terminator unconditional branch.
  FCmpBB->back().eraseFromParent();
  BasicBlock *FailBB =
      BasicBlock::Create(*Context, "", FCmpBB->getParent(), NextBB);

  // Create the shadow fcmp and comparison between the fcmps.
  IRBuilder<> FCmpBuilder(FCmpBB);
  FCmpBuilder.SetCurrentDebugLocation(FCmp.getDebugLoc());
  Value *ShadowLHS = Map.getShadow(LHS);
  Value *ShadowRHS = Map.getShadow(RHS);
  // See comment on ClTruncateFCmpEq.
  if (FCmp.isEquality() && ClTruncateFCmpEq) {
    Type *Ty = ShadowLHS->getType();
    ShadowLHS = FCmpBuilder.CreateCast(
        Instruction::FPExt,
        FCmpBuilder.CreateCast(Instruction::FPTrunc, ShadowLHS, LHS->getType()),
        Ty);
    ShadowRHS = FCmpBuilder.CreateCast(
        Instruction::FPExt,
        FCmpBuilder.CreateCast(Instruction::FPTrunc, ShadowRHS, RHS->getType()),
        Ty);
  }
  Value *ShadowFCmp =
      FCmpBuilder.CreateFCmp(FCmp.getPredicate(), ShadowLHS, ShadowRHS);
  Value *OriginalAndShadowFcmpMatch =
      FCmpBuilder.CreateICmpEQ(&FCmp, ShadowFCmp);

  if (OriginalAndShadowFcmpMatch->getType()->isVectorTy()) {
    // If we have a vector type, `OriginalAndShadowFcmpMatch` is a vector of i1,
    // where an element is true if the corresponding elements in original and
    // shadow are the same. We want all elements to be 1.
    OriginalAndShadowFcmpMatch =
        FCmpBuilder.CreateAndReduce(OriginalAndShadowFcmpMatch);
  }

  FCmpBuilder.CreateCondBr(OriginalAndShadowFcmpMatch, NextBB, FailBB);

  // Fill in FailBB.
  IRBuilder<> FailBuilder(FailBB);
  FailBuilder.SetCurrentDebugLocation(FCmp.getDebugLoc());

  const auto EmitFailCall = [this, &FCmp, &FCmpBuilder,
                             &FailBuilder](Value *L, Value *R, Value *ShadowL,
                                           Value *ShadowR, Value *Result,
                                           Value *ShadowResult) {
    Type *FT = L->getType();
    FunctionCallee *Callee = nullptr;
    if (FT->isFloatTy()) {
      Callee = &(NsanFCmpFail[kFloat]);
    } else if (FT->isDoubleTy()) {
      Callee = &(NsanFCmpFail[kDouble]);
    } else if (FT->isX86_FP80Ty()) {
      // FIXME: make NsanFCmpFailLongDouble work.
      Callee = &(NsanFCmpFail[kDouble]);
      L = FailBuilder.CreateCast(Instruction::FPTrunc, L,
                                 Type::getDoubleTy(*Context));
      R = FailBuilder.CreateCast(Instruction::FPTrunc, L,
                                 Type::getDoubleTy(*Context));
    } else {
      llvm_unreachable("not implemented");
    }
    FailBuilder.CreateCall(*Callee, {L, R, ShadowL, ShadowR,
                                     ConstantInt::get(FCmpBuilder.getInt32Ty(),
                                                      FCmp.getPredicate()),
                                     Result, ShadowResult});
  };
  if (LHS->getType()->isVectorTy()) {
    for (int I = 0, E = cast<VectorType>(LHS->getType())
                            ->getElementCount()
                            .getFixedValue();
         I < E; ++I) {
      EmitFailCall(FailBuilder.CreateExtractElement(LHS, I),
                   FailBuilder.CreateExtractElement(RHS, I),
                   FailBuilder.CreateExtractElement(ShadowLHS, I),
                   FailBuilder.CreateExtractElement(ShadowRHS, I),
                   FailBuilder.CreateExtractElement(&FCmp, I),
                   FailBuilder.CreateExtractElement(ShadowFCmp, I));
    }
  } else {
    EmitFailCall(LHS, RHS, ShadowLHS, ShadowRHS, &FCmp, ShadowFCmp);
  }
  FailBuilder.CreateBr(NextBB);

  ++NumInstrumentedFCmp;
}

// Creates a shadow phi value for any phi that defines a value of FT type.
PHINode *NumericalStabilitySanitizer::maybeCreateShadowPhi(
    PHINode &Phi, const TargetLibraryInfo &TLI) {
  Type *const VT = Phi.getType();
  Type *const ExtendedVT = Config.getExtendedFPType(VT);
  if (ExtendedVT == nullptr)
    return nullptr; // Not an FT value.
  // The phi operands are shadow values and are not available when the phi is
  // created. They will be populated in a final phase, once all shadow values
  // have been created.
  PHINode *Shadow = PHINode::Create(ExtendedVT, Phi.getNumIncomingValues());
  Shadow->insertAfter(&Phi);
  return Shadow;
}

Value *NumericalStabilitySanitizer::handleLoad(LoadInst &Load, Type *VT,
                                               Type *ExtendedVT) {
  IRBuilder<> Builder(getNextInstructionOrDie(Load));
  Builder.SetCurrentDebugLocation(Load.getDebugLoc());
  if (addrPointsToConstantData(Load.getPointerOperand())) {
    // No need to look into the shadow memory, the value is a constant. Just
    // convert from FT to 2FT.
    return Builder.CreateCast(Instruction::FPExt, &Load, ExtendedVT);
  }

  // if (%shadowptr == &)
  //    %shadow = fpext %v
  // else
  //    %shadow = load (ptrcast %shadow_ptr))
  // Considered options here:
  //  - Have `NsanGetShadowPtrForLoad` return a fixed address
  //    &__nsan_unknown_value_shadow_address that is valid to load from, and
  //    use a select. This has the advantage that the generated IR is simpler.
  //  - Have `NsanGetShadowPtrForLoad` return nullptr.  Because `select` does
  //    not short-circuit, dereferencing the returned pointer is no longer an
  //    option, have to split and create a separate basic block. This has the
  //    advantage of being easier to debug because it crashes if we ever mess
  //    up.

  const auto Extents = getMemoryExtentsOrDie(VT);
  Value *ShadowPtr = Builder.CreateCall(
      NsanGetShadowPtrForLoad[Extents.ValueType],
      {Load.getPointerOperand(), ConstantInt::get(IntptrTy, Extents.NumElts)});
  ++NumInstrumentedFTLoads;

  // Split the basic block.
  BasicBlock *LoadBB = Load.getParent();
  BasicBlock *NextBB = LoadBB->splitBasicBlock(Builder.GetInsertPoint());
  // Create the two options for creating the shadow value.
  BasicBlock *ShadowLoadBB =
      BasicBlock::Create(*Context, "", LoadBB->getParent(), NextBB);
  BasicBlock *FExtBB =
      BasicBlock::Create(*Context, "", LoadBB->getParent(), NextBB);

  // Replace the newly created terminator unconditional branch by a conditional
  // branch to one of the options.
  {
    LoadBB->back().eraseFromParent();
    IRBuilder<> LoadBBBuilder(LoadBB); // The old builder has been invalidated.
    LoadBBBuilder.SetCurrentDebugLocation(Load.getDebugLoc());
    LoadBBBuilder.CreateCondBr(LoadBBBuilder.CreateIsNull(ShadowPtr), FExtBB,
                               ShadowLoadBB);
  }

  // Fill in ShadowLoadBB.
  IRBuilder<> ShadowLoadBBBuilder(ShadowLoadBB);
  ShadowLoadBBBuilder.SetCurrentDebugLocation(Load.getDebugLoc());
  Value *ShadowLoad = ShadowLoadBBBuilder.CreateAlignedLoad(
      ExtendedVT, ShadowPtr, Align(1), Load.isVolatile());
  if (ClCheckLoads) {
    ShadowLoad = emitCheck(&Load, ShadowLoad, ShadowLoadBBBuilder,
                           CheckLoc::makeLoad(Load.getPointerOperand()));
  }
  ShadowLoadBBBuilder.CreateBr(NextBB);

  // Fill in FExtBB.
  IRBuilder<> FExtBBBuilder(FExtBB);
  FExtBBBuilder.SetCurrentDebugLocation(Load.getDebugLoc());
  Value *const FExt =
      FExtBBBuilder.CreateCast(Instruction::FPExt, &Load, ExtendedVT);
  FExtBBBuilder.CreateBr(NextBB);

  // The shadow value come from any of the options.
  IRBuilder<> NextBBBuilder(&*NextBB->begin());
  NextBBBuilder.SetCurrentDebugLocation(Load.getDebugLoc());
  PHINode *ShadowPhi = NextBBBuilder.CreatePHI(ExtendedVT, 2);
  ShadowPhi->addIncoming(ShadowLoad, ShadowLoadBB);
  ShadowPhi->addIncoming(FExt, FExtBB);
  return ShadowPhi;
}

Value *NumericalStabilitySanitizer::handleTrunc(FPTruncInst &Trunc, Type *VT,
                                                Type *ExtendedVT,
                                                const ValueToShadowMap &Map) {
  Value *const OrigSource = Trunc.getOperand(0);
  Type *const OrigSourceTy = OrigSource->getType();
  Type *const ExtendedSourceTy = Config.getExtendedFPType(OrigSourceTy);

  // When truncating:
  //  - (A) If the source has a shadow, we truncate from the shadow, else we
  //    truncate from the original source.
  //  - (B) If the shadow of the source is larger than the shadow of the dest,
  //    we still need a truncate. Else, the shadow of the source is the same
  //    type as the shadow of the dest (because mappings are non-decreasing), so
  //   we don't need to emit a truncate.
  // Examples,
  //   with a mapping of {f32->f64;f64->f80;f80->f128}
  //     fptrunc double   %1 to float     ->  fptrunc x86_fp80 s(%1) to double
  //     fptrunc x86_fp80 %1 to float     ->  fptrunc fp128    s(%1) to double
  //     fptrunc fp128    %1 to float     ->  fptrunc fp128    %1    to double
  //     fptrunc x86_fp80 %1 to double    ->  x86_fp80 s(%1)
  //     fptrunc fp128    %1 to double    ->  fptrunc fp128 %1 to x86_fp80
  //     fptrunc fp128    %1 to x86_fp80  ->  fp128 %1
  //   with a mapping of {f32->f64;f64->f128;f80->f128}
  //     fptrunc double   %1 to float     ->  fptrunc fp128    s(%1) to double
  //     fptrunc x86_fp80 %1 to float     ->  fptrunc fp128    s(%1) to double
  //     fptrunc fp128    %1 to float     ->  fptrunc fp128    %1    to double
  //     fptrunc x86_fp80 %1 to double    ->  fp128 %1
  //     fptrunc fp128    %1 to double    ->  fp128 %1
  //     fptrunc fp128    %1 to x86_fp80  ->  fp128 %1
  //   with a mapping of {f32->f32;f64->f32;f80->f64}
  //     fptrunc double   %1 to float     ->  float s(%1)
  //     fptrunc x86_fp80 %1 to float     ->  fptrunc double    s(%1) to float
  //     fptrunc fp128    %1 to float     ->  fptrunc fp128     %1    to float
  //     fptrunc x86_fp80 %1 to double    ->  fptrunc double    s(%1) to float
  //     fptrunc fp128    %1 to double    ->  fptrunc fp128     %1    to float
  //     fptrunc fp128    %1 to x86_fp80  ->  fptrunc fp128     %1    to double

  // See (A) above.
  Value *const Source =
      ExtendedSourceTy ? Map.getShadow(OrigSource) : OrigSource;
  Type *const SourceTy = ExtendedSourceTy ? ExtendedSourceTy : OrigSourceTy;
  // See (B) above.
  if (SourceTy == ExtendedVT)
    return Source;

  Instruction *const Shadow =
      CastInst::Create(Instruction::FPTrunc, Source, ExtendedVT);
  Shadow->insertAfter(&Trunc);
  return Shadow;
}

Value *NumericalStabilitySanitizer::handleExt(FPExtInst &Ext, Type *VT,
                                              Type *ExtendedVT,
                                              const ValueToShadowMap &Map) {
  Value *const OrigSource = Ext.getOperand(0);
  Type *const OrigSourceTy = OrigSource->getType();
  Type *const ExtendedSourceTy = Config.getExtendedFPType(OrigSourceTy);
  // When extending:
  //  - (A) If the source has a shadow, we extend from the shadow, else we
  //    extend from the original source.
  //  - (B) If the shadow of the dest is larger than the shadow of the source,
  //    we still need an extend. Else, the shadow of the source is the same
  //    type as the shadow of the dest (because mappings are non-decreasing), so
  //    we don't need to emit an extend.
  // Examples,
  //   with a mapping of {f32->f64;f64->f80;f80->f128}
  //     fpext half    %1 to float     ->  fpext half     %1    to double
  //     fpext half    %1 to double    ->  fpext half     %1    to x86_fp80
  //     fpext half    %1 to x86_fp80  ->  fpext half     %1    to fp128
  //     fpext float   %1 to double    ->  double s(%1)
  //     fpext float   %1 to x86_fp80  ->  fpext double   s(%1) to fp128
  //     fpext double  %1 to x86_fp80  ->  fpext x86_fp80 s(%1) to fp128
  //   with a mapping of {f32->f64;f64->f128;f80->f128}
  //     fpext half    %1 to float     ->  fpext half     %1    to double
  //     fpext half    %1 to double    ->  fpext half     %1    to fp128
  //     fpext half    %1 to x86_fp80  ->  fpext half     %1    to fp128
  //     fpext float   %1 to double    ->  fpext double   s(%1) to fp128
  //     fpext float   %1 to x86_fp80  ->  fpext double   s(%1) to fp128
  //     fpext double  %1 to x86_fp80  ->  fp128 s(%1)
  //   with a mapping of {f32->f32;f64->f32;f80->f64}
  //     fpext half    %1 to float     ->  fpext half     %1    to float
  //     fpext half    %1 to double    ->  fpext half     %1    to float
  //     fpext half    %1 to x86_fp80  ->  fpext half     %1    to double
  //     fpext float   %1 to double    ->  s(%1)
  //     fpext float   %1 to x86_fp80  ->  fpext float    s(%1) to double
  //     fpext double  %1 to x86_fp80  ->  fpext float    s(%1) to double

  // See (A) above.
  Value *const Source =
      ExtendedSourceTy ? Map.getShadow(OrigSource) : OrigSource;
  Type *const SourceTy = ExtendedSourceTy ? ExtendedSourceTy : OrigSourceTy;
  // See (B) above.
  if (SourceTy == ExtendedVT)
    return Source;

  Instruction *const Shadow =
      CastInst::Create(Instruction::FPExt, Source, ExtendedVT);
  Shadow->insertAfter(&Ext);
  return Shadow;
}

// Returns a value with the address of the callee.
Value *
NumericalStabilitySanitizer::getCalleeAddress(CallBase &Call,
                                              IRBuilder<> &Builder) const {
  if (Function *Fn = Call.getCalledFunction()) {
    // We're calling a statically known function.
    return Builder.CreatePtrToInt(Fn, IntptrTy);
  } else {
    // We're calling a function through a function pointer.
    return Builder.CreatePtrToInt(Call.getCalledOperand(), IntptrTy);
  }
}

namespace {

// FIXME: This should be tablegen-ed.

struct KnownIntrinsic {
  struct WidenedIntrinsic {
    const char *NarrowName;
    Intrinsic::ID ID; // wide id.
    using FnTypeFactory = FunctionType *(*)(LLVMContext &);
    FnTypeFactory MakeFnTy;
  };

  static const char *get(LibFunc LFunc);

  // Given an intrinsic with an `FT` argument, try to find a wider intrinsic
  // that applies the same operation on the shadow argument.
  // Options are:
  //  - pass in the ID and full function type,
  //  - pass in the name, which includes the function type through mangling.
  static const WidenedIntrinsic *widen(StringRef Name);

private:
  struct LFEntry {
    LibFunc LFunc;
    const char *IntrinsicName;
  };
  static const LFEntry kLibfuncIntrinsics[];

  static const WidenedIntrinsic kWidenedIntrinsics[];
};

FunctionType *Make_Double_Double(LLVMContext &C) {
  return FunctionType::get(Type::getDoubleTy(C), {Type::getDoubleTy(C)}, false);
}

FunctionType *Make_X86FP80_X86FP80(LLVMContext &C) {
  return FunctionType::get(Type::getX86_FP80Ty(C), {Type::getX86_FP80Ty(C)},
                           false);
}

FunctionType *Make_Double_DoubleI32(LLVMContext &C) {
  return FunctionType::get(Type::getDoubleTy(C),
                           {Type::getDoubleTy(C), Type::getInt32Ty(C)}, false);
}

FunctionType *Make_X86FP80_X86FP80I32(LLVMContext &C) {
  return FunctionType::get(Type::getX86_FP80Ty(C),
                           {Type::getX86_FP80Ty(C), Type::getInt32Ty(C)},
                           false);
}

FunctionType *Make_Double_DoubleDouble(LLVMContext &C) {
  return FunctionType::get(Type::getDoubleTy(C),
                           {Type::getDoubleTy(C), Type::getDoubleTy(C)}, false);
}

FunctionType *Make_X86FP80_X86FP80X86FP80(LLVMContext &C) {
  return FunctionType::get(Type::getX86_FP80Ty(C),
                           {Type::getX86_FP80Ty(C), Type::getX86_FP80Ty(C)},
                           false);
}

FunctionType *Make_Double_DoubleDoubleDouble(LLVMContext &C) {
  return FunctionType::get(
      Type::getDoubleTy(C),
      {Type::getDoubleTy(C), Type::getDoubleTy(C), Type::getDoubleTy(C)},
      false);
}

FunctionType *Make_X86FP80_X86FP80X86FP80X86FP80(LLVMContext &C) {
  return FunctionType::get(
      Type::getX86_FP80Ty(C),
      {Type::getX86_FP80Ty(C), Type::getX86_FP80Ty(C), Type::getX86_FP80Ty(C)},
      false);
}

const KnownIntrinsic::WidenedIntrinsic KnownIntrinsic::kWidenedIntrinsics[] = {
    // FIXME: Right now we ignore vector intrinsics.
    // This is hard because we have to model the semantics of the intrinsics,
    // e.g. llvm.x86.sse2.min.sd means extract first element, min, insert back.
    // Intrinsics that take any non-vector FT types:
    // NOTE: Right now because of https://bugs.llvm.org/show_bug.cgi?id=45399
    // for f128 we need to use Make_X86FP80_X86FP80 (go to a lower precision and
    // come back).
    {"llvm.sqrt.f32", Intrinsic::sqrt, Make_Double_Double},
    {"llvm.sqrt.f64", Intrinsic::sqrt, Make_X86FP80_X86FP80},
    {"llvm.sqrt.f80", Intrinsic::sqrt, Make_X86FP80_X86FP80},
    {"llvm.powi.f32", Intrinsic::powi, Make_Double_DoubleI32},
    {"llvm.powi.f64", Intrinsic::powi, Make_X86FP80_X86FP80I32},
    {"llvm.powi.f80", Intrinsic::powi, Make_X86FP80_X86FP80I32},
    {"llvm.sin.f32", Intrinsic::sin, Make_Double_Double},
    {"llvm.sin.f64", Intrinsic::sin, Make_X86FP80_X86FP80},
    {"llvm.sin.f80", Intrinsic::sin, Make_X86FP80_X86FP80},
    {"llvm.cos.f32", Intrinsic::cos, Make_Double_Double},
    {"llvm.cos.f64", Intrinsic::cos, Make_X86FP80_X86FP80},
    {"llvm.cos.f80", Intrinsic::cos, Make_X86FP80_X86FP80},
    {"llvm.pow.f32", Intrinsic::pow, Make_Double_DoubleDouble},
    {"llvm.pow.f64", Intrinsic::pow, Make_X86FP80_X86FP80X86FP80},
    {"llvm.pow.f80", Intrinsic::pow, Make_X86FP80_X86FP80X86FP80},
    {"llvm.exp.f32", Intrinsic::exp, Make_Double_Double},
    {"llvm.exp.f64", Intrinsic::exp, Make_X86FP80_X86FP80},
    {"llvm.exp.f80", Intrinsic::exp, Make_X86FP80_X86FP80},
    {"llvm.exp2.f32", Intrinsic::exp2, Make_Double_Double},
    {"llvm.exp2.f64", Intrinsic::exp2, Make_X86FP80_X86FP80},
    {"llvm.exp2.f80", Intrinsic::exp2, Make_X86FP80_X86FP80},
    {"llvm.log.f32", Intrinsic::log, Make_Double_Double},
    {"llvm.log.f64", Intrinsic::log, Make_X86FP80_X86FP80},
    {"llvm.log.f80", Intrinsic::log, Make_X86FP80_X86FP80},
    {"llvm.log10.f32", Intrinsic::log10, Make_Double_Double},
    {"llvm.log10.f64", Intrinsic::log10, Make_X86FP80_X86FP80},
    {"llvm.log10.f80", Intrinsic::log10, Make_X86FP80_X86FP80},
    {"llvm.log2.f32", Intrinsic::log2, Make_Double_Double},
    {"llvm.log2.f64", Intrinsic::log2, Make_X86FP80_X86FP80},
    {"llvm.log2.f80", Intrinsic::log2, Make_X86FP80_X86FP80},
    {"llvm.fma.f32", Intrinsic::fma, Make_Double_DoubleDoubleDouble},

    {"llvm.fmuladd.f32", Intrinsic::fmuladd, Make_Double_DoubleDoubleDouble},

    {"llvm.fma.f64", Intrinsic::fma, Make_X86FP80_X86FP80X86FP80X86FP80},

    {"llvm.fmuladd.f64", Intrinsic::fma, Make_X86FP80_X86FP80X86FP80X86FP80},

    {"llvm.fma.f80", Intrinsic::fma, Make_X86FP80_X86FP80X86FP80X86FP80},
    {"llvm.fabs.f32", Intrinsic::fabs, Make_Double_Double},
    {"llvm.fabs.f64", Intrinsic::fabs, Make_X86FP80_X86FP80},
    {"llvm.fabs.f80", Intrinsic::fabs, Make_X86FP80_X86FP80},
    {"llvm.minnum.f32", Intrinsic::minnum, Make_Double_DoubleDouble},
    {"llvm.minnum.f64", Intrinsic::minnum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.minnum.f80", Intrinsic::minnum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.maxnum.f32", Intrinsic::maxnum, Make_Double_DoubleDouble},
    {"llvm.maxnum.f64", Intrinsic::maxnum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.maxnum.f80", Intrinsic::maxnum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.minimum.f32", Intrinsic::minimum, Make_Double_DoubleDouble},
    {"llvm.minimum.f64", Intrinsic::minimum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.minimum.f80", Intrinsic::minimum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.maximum.f32", Intrinsic::maximum, Make_Double_DoubleDouble},
    {"llvm.maximum.f64", Intrinsic::maximum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.maximum.f80", Intrinsic::maximum, Make_X86FP80_X86FP80X86FP80},
    {"llvm.copysign.f32", Intrinsic::copysign, Make_Double_DoubleDouble},
    {"llvm.copysign.f64", Intrinsic::copysign, Make_X86FP80_X86FP80X86FP80},
    {"llvm.copysign.f80", Intrinsic::copysign, Make_X86FP80_X86FP80X86FP80},
    {"llvm.floor.f32", Intrinsic::floor, Make_Double_Double},
    {"llvm.floor.f64", Intrinsic::floor, Make_X86FP80_X86FP80},
    {"llvm.floor.f80", Intrinsic::floor, Make_X86FP80_X86FP80},
    {"llvm.ceil.f32", Intrinsic::ceil, Make_Double_Double},
    {"llvm.ceil.f64", Intrinsic::ceil, Make_X86FP80_X86FP80},
    {"llvm.ceil.f80", Intrinsic::ceil, Make_X86FP80_X86FP80},
    {"llvm.trunc.f32", Intrinsic::trunc, Make_Double_Double},
    {"llvm.trunc.f64", Intrinsic::trunc, Make_X86FP80_X86FP80},
    {"llvm.trunc.f80", Intrinsic::trunc, Make_X86FP80_X86FP80},
    {"llvm.rint.f32", Intrinsic::rint, Make_Double_Double},
    {"llvm.rint.f64", Intrinsic::rint, Make_X86FP80_X86FP80},
    {"llvm.rint.f80", Intrinsic::rint, Make_X86FP80_X86FP80},
    {"llvm.nearbyint.f32", Intrinsic::nearbyint, Make_Double_Double},
    {"llvm.nearbyint.f64", Intrinsic::nearbyint, Make_X86FP80_X86FP80},
    {"llvm.nearbyin80f64", Intrinsic::nearbyint, Make_X86FP80_X86FP80},
    {"llvm.round.f32", Intrinsic::round, Make_Double_Double},
    {"llvm.round.f64", Intrinsic::round, Make_X86FP80_X86FP80},
    {"llvm.round.f80", Intrinsic::round, Make_X86FP80_X86FP80},
    {"llvm.lround.f32", Intrinsic::lround, Make_Double_Double},
    {"llvm.lround.f64", Intrinsic::lround, Make_X86FP80_X86FP80},
    {"llvm.lround.f80", Intrinsic::lround, Make_X86FP80_X86FP80},
    {"llvm.llround.f32", Intrinsic::llround, Make_Double_Double},
    {"llvm.llround.f64", Intrinsic::llround, Make_X86FP80_X86FP80},
    {"llvm.llround.f80", Intrinsic::llround, Make_X86FP80_X86FP80},
    {"llvm.lrint.f32", Intrinsic::lrint, Make_Double_Double},
    {"llvm.lrint.f64", Intrinsic::lrint, Make_X86FP80_X86FP80},
    {"llvm.lrint.f80", Intrinsic::lrint, Make_X86FP80_X86FP80},
    {"llvm.llrint.f32", Intrinsic::llrint, Make_Double_Double},
    {"llvm.llrint.f64", Intrinsic::llrint, Make_X86FP80_X86FP80},
    {"llvm.llrint.f80", Intrinsic::llrint, Make_X86FP80_X86FP80},
};

const KnownIntrinsic::LFEntry KnownIntrinsic::kLibfuncIntrinsics[] = {
    {LibFunc_sqrtf, "llvm.sqrt.f32"},           //
    {LibFunc_sqrt, "llvm.sqrt.f64"},            //
    {LibFunc_sqrtl, "llvm.sqrt.f80"},           //
    {LibFunc_sinf, "llvm.sin.f32"},             //
    {LibFunc_sin, "llvm.sin.f64"},              //
    {LibFunc_sinl, "llvm.sin.f80"},             //
    {LibFunc_cosf, "llvm.cos.f32"},             //
    {LibFunc_cos, "llvm.cos.f64"},              //
    {LibFunc_cosl, "llvm.cos.f80"},             //
    {LibFunc_powf, "llvm.pow.f32"},             //
    {LibFunc_pow, "llvm.pow.f64"},              //
    {LibFunc_powl, "llvm.pow.f80"},             //
    {LibFunc_expf, "llvm.exp.f32"},             //
    {LibFunc_exp, "llvm.exp.f64"},              //
    {LibFunc_expl, "llvm.exp.f80"},             //
    {LibFunc_exp2f, "llvm.exp2.f32"},           //
    {LibFunc_exp2, "llvm.exp2.f64"},            //
    {LibFunc_exp2l, "llvm.exp2.f80"},           //
    {LibFunc_logf, "llvm.log.f32"},             //
    {LibFunc_log, "llvm.log.f64"},              //
    {LibFunc_logl, "llvm.log.f80"},             //
    {LibFunc_log10f, "llvm.log10.f32"},         //
    {LibFunc_log10, "llvm.log10.f64"},          //
    {LibFunc_log10l, "llvm.log10.f80"},         //
    {LibFunc_log2f, "llvm.log2.f32"},           //
    {LibFunc_log2, "llvm.log2.f64"},            //
    {LibFunc_log2l, "llvm.log2.f80"},           //
    {LibFunc_fabsf, "llvm.fabs.f32"},           //
    {LibFunc_fabs, "llvm.fabs.f64"},            //
    {LibFunc_fabsl, "llvm.fabs.f80"},           //
    {LibFunc_copysignf, "llvm.copysign.f32"},   //
    {LibFunc_copysign, "llvm.copysign.f64"},    //
    {LibFunc_copysignl, "llvm.copysign.f80"},   //
    {LibFunc_floorf, "llvm.floor.f32"},         //
    {LibFunc_floor, "llvm.floor.f64"},          //
    {LibFunc_floorl, "llvm.floor.f80"},         //
    {LibFunc_fmaxf, "llvm.maxnum.f32"},         //
    {LibFunc_fmax, "llvm.maxnum.f64"},          //
    {LibFunc_fmaxl, "llvm.maxnum.f80"},         //
    {LibFunc_fminf, "llvm.minnum.f32"},         //
    {LibFunc_fmin, "llvm.minnum.f64"},          //
    {LibFunc_fminl, "llvm.minnum.f80"},         //
    {LibFunc_ceilf, "llvm.ceil.f32"},           //
    {LibFunc_ceil, "llvm.ceil.f64"},            //
    {LibFunc_ceill, "llvm.ceil.f80"},           //
    {LibFunc_truncf, "llvm.trunc.f32"},         //
    {LibFunc_trunc, "llvm.trunc.f64"},          //
    {LibFunc_truncl, "llvm.trunc.f80"},         //
    {LibFunc_rintf, "llvm.rint.f32"},           //
    {LibFunc_rint, "llvm.rint.f64"},            //
    {LibFunc_rintl, "llvm.rint.f80"},           //
    {LibFunc_nearbyintf, "llvm.nearbyint.f32"}, //
    {LibFunc_nearbyint, "llvm.nearbyint.f64"},  //
    {LibFunc_nearbyintl, "llvm.nearbyint.f80"}, //
    {LibFunc_roundf, "llvm.round.f32"},         //
    {LibFunc_round, "llvm.round.f64"},          //
    {LibFunc_roundl, "llvm.round.f80"},         //
};

const char *KnownIntrinsic::get(LibFunc LFunc) {
  for (const auto &E : kLibfuncIntrinsics) {
    if (E.LFunc == LFunc)
      return E.IntrinsicName;
  }
  return nullptr;
}

const KnownIntrinsic::WidenedIntrinsic *KnownIntrinsic::widen(StringRef Name) {
  for (const auto &E : kWidenedIntrinsics) {
    if (E.NarrowName == Name)
      return &E;
  }
  return nullptr;
}

} // namespace

// Returns the name of the LLVM intrinsic corresponding to the given function.
static const char *getIntrinsicFromLibfunc(Function &Fn, Type *VT,
                                           const TargetLibraryInfo &TLI) {
  LibFunc LFunc;
  if (!TLI.getLibFunc(Fn, LFunc))
    return nullptr;

  if (const char *Name = KnownIntrinsic::get(LFunc))
    return Name;

  errs() << "FIXME: LibFunc: " << TLI.getName(LFunc) << "\n";
  return nullptr;
}

// Try to handle a known function call.
Value *NumericalStabilitySanitizer::maybeHandleKnownCallBase(
    CallBase &Call, Type *VT, Type *ExtendedVT, const TargetLibraryInfo &TLI,
    const ValueToShadowMap &Map, IRBuilder<> &Builder) {
  Function *const Fn = Call.getCalledFunction();
  if (Fn == nullptr)
    return nullptr;

  Intrinsic::ID WidenedId = Intrinsic::ID();
  FunctionType *WidenedFnTy = nullptr;
  if (const auto ID = Fn->getIntrinsicID()) {
    const auto *const Widened = KnownIntrinsic::widen(Fn->getName());
    if (Widened) {
      WidenedId = Widened->ID;
      WidenedFnTy = Widened->MakeFnTy(*Context);
    } else {
      // If we don't know how to widen the intrinsic, we have no choice but to
      // call the non-wide version on a truncated shadow and extend again
      // afterwards.
      WidenedId = ID;
      WidenedFnTy = Fn->getFunctionType();
    }
  } else if (const char *Name = getIntrinsicFromLibfunc(*Fn, VT, TLI)) {
    // We might have a call to a library function that we can replace with a
    // wider Intrinsic.
    const auto *Widened = KnownIntrinsic::widen(Name);
    assert(Widened && "make sure KnownIntrinsic entries are consistent");
    WidenedId = Widened->ID;
    WidenedFnTy = Widened->MakeFnTy(*Context);
  } else {
    // This is not a known library function or intrinsic.
    return nullptr;
  }

  // Check that the widened intrinsic is valid.
  SmallVector<Intrinsic::IITDescriptor, 8> Table;
  getIntrinsicInfoTableEntries(WidenedId, Table);
  SmallVector<Type *, 4> ArgTys;
  ArrayRef<Intrinsic::IITDescriptor> TableRef = Table;
  const Intrinsic::MatchIntrinsicTypesResult Res =
      Intrinsic::matchIntrinsicSignature(WidenedFnTy, TableRef, ArgTys);
  assert(Res == Intrinsic::MatchIntrinsicTypes_Match &&
         "invalid widened intrinsic");
  (void)Res;

  // For known intrinsic functions, we create a second call to the same
  // intrinsic with a different type.
  SmallVector<Value *, 4> Args;
  // The last operand is the intrinsic itself, skip it.
  for (unsigned I = 0, E = Call.getNumOperands() - 1; I < E; ++I) {
    Value *Arg = Call.getOperand(I);
    Type *const OrigArgTy = Arg->getType();
    Type *const IntrinsicArgTy = WidenedFnTy->getParamType(I);
    if (OrigArgTy == IntrinsicArgTy) {
      Args.push_back(Arg); // The arg is passed as is.
      continue;
    }
    Type *const ShadowArgTy = Config.getExtendedFPType(Arg->getType());
    assert(ShadowArgTy &&
           "don't know how to get the shadow value for a non-FT");
    Value *Shadow = Map.getShadow(Arg);
    if (ShadowArgTy == IntrinsicArgTy) {
      // The shadow is the right type for the intrinsic.
      assert(Shadow->getType() == ShadowArgTy);
      Args.push_back(Shadow);
      continue;
    }
    // There is no intrinsic with his level of precision, truncate the shadow.
    Args.push_back(
        Builder.CreateCast(Instruction::FPTrunc, Shadow, IntrinsicArgTy));
  }
  Value *IntrinsicCall = Builder.CreateIntrinsic(WidenedId, ArgTys, Args);
  return WidenedFnTy->getReturnType() == ExtendedVT
             ? IntrinsicCall
             : Builder.CreateCast(Instruction::FPExt, IntrinsicCall,
                                  ExtendedVT);
}

// Handle a CallBase, i.e. a function call, an inline asm sequence, or an
// invoke.
Value *NumericalStabilitySanitizer::handleCallBase(CallBase &Call, Type *VT,
                                                   Type *ExtendedVT,
                                                   const TargetLibraryInfo &TLI,
                                                   const ValueToShadowMap &Map,
                                                   IRBuilder<> &Builder) {
  // We cannot look inside inline asm, just expand the result again.
  if (Call.isInlineAsm()) {
    return Builder.CreateCast(Instruction::FPExt, &Call, ExtendedVT);
  }

  // Intrinsics and library functions (e.g. sin, exp) are handled
  // specifically, because we know their semantics and can do better than
  // blindly calling them (e.g. compute the sinus in the actual shadow domain).
  if (Value *V =
          maybeHandleKnownCallBase(Call, VT, ExtendedVT, TLI, Map, Builder))
    return V;

  // If the return tag matches that of the called function, read the extended
  // return value from the shadow ret ptr. Else, just extend the return value.
  Value *HasShadowRet = Builder.CreateICmpEQ(
      Builder.CreateLoad(IntptrTy, NsanShadowRetTag, /*isVolatile=*/false),
      getCalleeAddress(Call, Builder));

  Value *ShadowRetVal = Builder.CreateLoad(
      ExtendedVT,
      Builder.CreateConstGEP2_64(NsanShadowRetType, NsanShadowRetPtr, 0, 0),
      /*isVolatile=*/false);
  Value *Shadow = Builder.CreateSelect(
      HasShadowRet, ShadowRetVal,
      Builder.CreateCast(Instruction::FPExt, &Call, ExtendedVT));
  ++NumInstrumentedFTCalls;
  return Shadow;
  // Note that we do not need to set NsanShadowRetTag to zero as we know that
  // either the function is not instrumented and it will never set
  // NsanShadowRetTag; or it is and it will always do so.
}

// Creates a shadow value for the given FT value. At that point all operands are
// guaranteed to be available.
Value *NumericalStabilitySanitizer::createShadowValueWithOperandsAvailable(
    Instruction &Inst, const TargetLibraryInfo &TLI,
    const ValueToShadowMap &Map) {
  Type *const VT = Inst.getType();
  Type *const ExtendedVT = Config.getExtendedFPType(VT);
  assert(ExtendedVT != nullptr && "trying to create a shadow for a non-FT");

  if (LoadInst *Load = dyn_cast<LoadInst>(&Inst)) {
    return handleLoad(*Load, VT, ExtendedVT);
  }
  if (CallInst *Call = dyn_cast<CallInst>(&Inst)) {
    // Insert after the call.
    BasicBlock::iterator It(Inst);
    IRBuilder<> Builder(Call->getParent(), ++It);
    Builder.SetCurrentDebugLocation(Call->getDebugLoc());
    return handleCallBase(*Call, VT, ExtendedVT, TLI, Map, Builder);
  }
  if (InvokeInst *Invoke = dyn_cast<InvokeInst>(&Inst)) {
    // The Invoke terminates the basic block, create a new basic block in
    // between the successful invoke and the next block.
    BasicBlock *InvokeBB = Invoke->getParent();
    BasicBlock *NextBB = Invoke->getNormalDest();
    BasicBlock *NewBB =
        BasicBlock::Create(*Context, "", NextBB->getParent(), NextBB);
    Inst.replaceSuccessorWith(NextBB, NewBB);

    IRBuilder<> Builder(NewBB);
    Builder.SetCurrentDebugLocation(Invoke->getDebugLoc());
    Value *Shadow = handleCallBase(*Invoke, VT, ExtendedVT, TLI, Map, Builder);
    Builder.CreateBr(NextBB);
    NewBB->replaceSuccessorsPhiUsesWith(InvokeBB, NewBB);
    return Shadow;
  }
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*BinOp));
    Builder.SetCurrentDebugLocation(BinOp->getDebugLoc());
    return Builder.CreateBinOp(BinOp->getOpcode(),
                               Map.getShadow(BinOp->getOperand(0)),
                               Map.getShadow(BinOp->getOperand(1)));
  }
  if (UnaryOperator *UnaryOp = dyn_cast<UnaryOperator>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*UnaryOp));
    Builder.SetCurrentDebugLocation(UnaryOp->getDebugLoc());
    return Builder.CreateUnOp(UnaryOp->getOpcode(),
                              Map.getShadow(UnaryOp->getOperand(0)));
  }
  if (FPTruncInst *Trunc = dyn_cast<FPTruncInst>(&Inst)) {
    return handleTrunc(*Trunc, VT, ExtendedVT, Map);
  }
  if (FPExtInst *Ext = dyn_cast<FPExtInst>(&Inst)) {
    return handleExt(*Ext, VT, ExtendedVT, Map);
  }
  if (isa<UIToFPInst>(&Inst) || isa<SIToFPInst>(&Inst)) {
    CastInst *Cast = dyn_cast<CastInst>(&Inst);
    IRBuilder<> Builder(getNextInstructionOrDie(*Cast));
    Builder.SetCurrentDebugLocation(Cast->getDebugLoc());
    return Builder.CreateCast(Cast->getOpcode(), Cast->getOperand(0),
                              ExtendedVT);
  }

  if (SelectInst *S = dyn_cast<SelectInst>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*S));
    Builder.SetCurrentDebugLocation(S->getDebugLoc());
    return Builder.CreateSelect(S->getCondition(),
                                Map.getShadow(S->getTrueValue()),
                                Map.getShadow(S->getFalseValue()));
  }

  if (ExtractElementInst *Extract = dyn_cast<ExtractElementInst>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*Extract));
    Builder.SetCurrentDebugLocation(Extract->getDebugLoc());
    return Builder.CreateExtractElement(
        Map.getShadow(Extract->getVectorOperand()), Extract->getIndexOperand());
  }

  if (InsertElementInst *Insert = dyn_cast<InsertElementInst>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*Insert));
    Builder.SetCurrentDebugLocation(Insert->getDebugLoc());
    return Builder.CreateInsertElement(Map.getShadow(Insert->getOperand(0)),
                                       Map.getShadow(Insert->getOperand(1)),
                                       Insert->getOperand(2));
  }

  if (ShuffleVectorInst *Shuffle = dyn_cast<ShuffleVectorInst>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*Shuffle));
    Builder.SetCurrentDebugLocation(Shuffle->getDebugLoc());
    return Builder.CreateShuffleVector(Map.getShadow(Shuffle->getOperand(0)),
                                       Map.getShadow(Shuffle->getOperand(1)),
                                       Shuffle->getShuffleMask());
  }

  if (ExtractValueInst *Extract = dyn_cast<ExtractValueInst>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*Extract));
    Builder.SetCurrentDebugLocation(Extract->getDebugLoc());
    // FIXME: We could make aggregate object first class citizens. For now we
    // just extend the extracted value.
    return Builder.CreateCast(Instruction::FPExt, Extract, ExtendedVT);
  }

  if (BitCastInst *BC = dyn_cast<BitCastInst>(&Inst)) {
    IRBuilder<> Builder(getNextInstructionOrDie(*BC));
    Builder.SetCurrentDebugLocation(BC->getDebugLoc());
    return Builder.CreateCast(Instruction::FPExt, BC, ExtendedVT);
  }

  errs() << "FIXME: implement " << Inst.getOpcodeName() << "\n";
  llvm_unreachable("not implemented");
}

// Creates a shadow value for an instruction that defines a value of FT type.
// FT operands that do not already have shadow values are created recursively.
// The DFS is guaranteed to not loop as phis and arguments already have
// shadows.
void NumericalStabilitySanitizer::maybeCreateShadowValue(
    Instruction &Root, const TargetLibraryInfo &TLI, ValueToShadowMap &Map) {
  Type *const VT = Root.getType();
  Type *const ExtendedVT = Config.getExtendedFPType(VT);
  if (ExtendedVT == nullptr)
    return; // Not an FT value.

  if (Map.hasShadow(&Root))
    return; // Shadow already exists.

  assert(!isa<PHINode>(Root) && "phi nodes should already have shadows");

  std::vector<Instruction *> DfsStack(1, &Root);
  while (!DfsStack.empty()) {
    // Ensure that all operands to the instruction have shadows before
    // proceeding.
    Instruction *I = DfsStack.back();
    // The shadow for the instruction might have been created deeper in the DFS,
    // see `forward_use_with_two_uses` test.
    if (Map.hasShadow(I)) {
      DfsStack.pop_back();
      continue;
    }

    bool MissingShadow = false;
    for (Value *Op : I->operands()) {
      Type *const VT = Op->getType();
      if (!Config.getExtendedFPType(VT))
        continue; // Not an FT value.
      if (Map.hasShadow(Op))
        continue; // Shadow is already available.
      assert(isa<Instruction>(Op) &&
             "non-instructions should already have shadows");
      assert(!isa<PHINode>(Op) && "phi nodes should aready have shadows");
      MissingShadow = true;
      DfsStack.push_back(dyn_cast<Instruction>(Op));
    }
    if (MissingShadow)
      continue; // Process operands and come back to this instruction later.

    // All operands have shadows. Create a shadow for the current value.
    Value *Shadow = createShadowValueWithOperandsAvailable(*I, TLI, Map);
    Map.setShadow(I, Shadow);
    DfsStack.pop_back();
  }
}

// A floating-point store needs its value and type written to shadow memory.
void NumericalStabilitySanitizer::propagateFTStore(
    StoreInst &Store, Type *const VT, Type *const ExtendedVT,
    const ValueToShadowMap &Map) {
  Value *StoredValue = Store.getValueOperand();
  IRBuilder<> Builder(&Store);
  Builder.SetCurrentDebugLocation(Store.getDebugLoc());
  const auto Extents = getMemoryExtentsOrDie(VT);
  Value *ShadowPtr = Builder.CreateCall(
      NsanGetShadowPtrForStore[Extents.ValueType],
      {Store.getPointerOperand(), ConstantInt::get(IntptrTy, Extents.NumElts)});

  Value *StoredShadow = Map.getShadow(StoredValue);
  if (!Store.getParent()->getParent()->hasOptNone()) {
    // Only check stores when optimizing, because non-optimized code generates
    // too many stores to the stack, creating false positives.
    if (ClCheckStores) {
      StoredShadow = emitCheck(StoredValue, StoredShadow, Builder,
                               CheckLoc::makeStore(Store.getPointerOperand()));
      ++NumInstrumentedFTStores;
    }
  }

  Builder.CreateAlignedStore(StoredShadow, ShadowPtr, Align(1),
                             Store.isVolatile());
}

// A non-ft store needs to invalidate shadow memory. Exceptions are:
//   - memory transfers of floating-point data through other pointer types (llvm
//     optimization passes transform `*(float*)a = *(float*)b` into
//     `*(i32*)a = *(i32*)b` ). These have the same semantics as memcpy.
//   - Writes of FT-sized constants. LLVM likes to do float stores as bitcasted
//     ints. Note that this is not really necessary because if the value is
//     unknown the framework will re-extend it on load anyway. It just felt
//     easier to debug tests with vectors of FTs.
void NumericalStabilitySanitizer::propagateNonFTStore(
    StoreInst &Store, Type *const VT, const ValueToShadowMap &Map) {
  Value *PtrOp = Store.getPointerOperand();
  IRBuilder<> Builder(getNextInstructionOrDie(Store));
  Builder.SetCurrentDebugLocation(Store.getDebugLoc());
  Value *Dst = PtrOp;
  const DataLayout &DL =
      Store.getParent()->getParent()->getParent()->getDataLayout();
  TypeSize SlotSize = DL.getTypeStoreSize(VT);
  assert(!SlotSize.isScalable() && "unsupported");
  const auto LoadSizeBytes = SlotSize.getFixedValue();
  Value *ValueSize = Builder.Insert(Constant::getIntegerValue(
      IntptrTy, APInt(IntptrTy->getPrimitiveSizeInBits(), LoadSizeBytes)));

  ++NumInstrumentedNonFTStores;
  Value *StoredValue = Store.getValueOperand();
  if (LoadInst *Load = dyn_cast<LoadInst>(StoredValue)) {
    // FIXME: Handle the case when the value is from a phi.
    // This is a memory transfer with memcpy semantics. Copy the type and
    // value from the source. Note that we cannot use __nsan_copy_values()
    // here, because that will not work when there is a write to memory in
    // between the load and the store, e.g. in the case of a swap.
    Type *ShadowTypeIntTy = Type::getIntNTy(*Context, 8 * LoadSizeBytes);
    Type *ShadowValueIntTy =
        Type::getIntNTy(*Context, 8 * kShadowScale * LoadSizeBytes);
    IRBuilder<> LoadBuilder(getNextInstructionOrDie(*Load));
    Builder.SetCurrentDebugLocation(Store.getDebugLoc());
    Value *LoadSrc = Load->getPointerOperand();
    // Read the shadow type and value at load time. The type has the same size
    // as the FT value, the value has twice its size.
    // FIXME: cache them to avoid re-creating them when a load is used by
    // several stores. Maybe create them like the FT shadows when a load is
    // encountered.
    Value *RawShadowType = LoadBuilder.CreateAlignedLoad(
        ShadowTypeIntTy,
        LoadBuilder.CreateCall(NsanGetRawShadowTypePtr, {LoadSrc}), Align(1),
        /*isVolatile=*/false);
    Value *RawShadowValue = LoadBuilder.CreateAlignedLoad(
        ShadowValueIntTy,
        LoadBuilder.CreateCall(NsanGetRawShadowPtr, {LoadSrc}), Align(1),
        /*isVolatile=*/false);

    // Write back the shadow type and value at store time.
    Builder.CreateAlignedStore(
        RawShadowType, Builder.CreateCall(NsanGetRawShadowTypePtr, {Dst}),
        Align(1),
        /*isVolatile=*/false);
    Builder.CreateAlignedStore(RawShadowValue,
                               Builder.CreateCall(NsanGetRawShadowPtr, {Dst}),
                               Align(1),
                               /*isVolatile=*/false);

    ++NumInstrumentedNonFTMemcpyStores;
    return;
  }
  if (Constant *C = dyn_cast<Constant>(StoredValue)) {
    // This might be a fp constant stored as an int. Bitcast and store if it has
    // appropriate size.
    Type *BitcastTy = nullptr; // The FT type to bitcast to.
    if (ConstantInt *CInt = dyn_cast<ConstantInt>(C)) {
      switch (CInt->getType()->getScalarSizeInBits()) {
      case 32:
        BitcastTy = Type::getFloatTy(*Context);
        break;
      case 64:
        BitcastTy = Type::getDoubleTy(*Context);
        break;
      case 80:
        BitcastTy = Type::getX86_FP80Ty(*Context);
        break;
      default:
        break;
      }
    } else if (ConstantDataVector *CDV = dyn_cast<ConstantDataVector>(C)) {
      const int NumElements =
          cast<VectorType>(CDV->getType())->getElementCount().getFixedValue();
      switch (CDV->getType()->getScalarSizeInBits()) {
      case 32:
        BitcastTy =
            VectorType::get(Type::getFloatTy(*Context), NumElements, false);
        break;
      case 64:
        BitcastTy =
            VectorType::get(Type::getDoubleTy(*Context), NumElements, false);
        break;
      case 80:
        BitcastTy =
            VectorType::get(Type::getX86_FP80Ty(*Context), NumElements, false);
        break;
      default:
        break;
      }
    }
    if (BitcastTy) {
      const MemoryExtents Extents = getMemoryExtentsOrDie(BitcastTy);
      Value *ShadowPtr = Builder.CreateCall(
          NsanGetShadowPtrForStore[Extents.ValueType],
          {PtrOp, ConstantInt::get(IntptrTy, Extents.NumElts)});
      // Bitcast the integer value to the appropriate FT type and extend to 2FT.
      Type *ExtVT = Config.getExtendedFPType(BitcastTy);
      Value *Shadow = Builder.CreateCast(
          Instruction::FPExt, Builder.CreateBitCast(C, BitcastTy), ExtVT);
      Builder.CreateAlignedStore(Shadow, ShadowPtr, Align(1),
                                 Store.isVolatile());
      return;
    }
  }
  // All other stores just reset the shadow value to unknown.
  Builder.CreateCall(NsanSetValueUnknown, {Dst, ValueSize});
}

void NumericalStabilitySanitizer::propagateShadowValues(
    Instruction &Inst, const TargetLibraryInfo &TLI,
    const ValueToShadowMap &Map) {
  if (StoreInst *Store = dyn_cast<StoreInst>(&Inst)) {
    Value *StoredValue = Store->getValueOperand();
    Type *const VT = StoredValue->getType();
    Type *const ExtendedVT = Config.getExtendedFPType(VT);
    if (ExtendedVT == nullptr)
      return propagateNonFTStore(*Store, VT, Map);
    return propagateFTStore(*Store, VT, ExtendedVT, Map);
  }

  if (FCmpInst *FCmp = dyn_cast<FCmpInst>(&Inst)) {
    emitFCmpCheck(*FCmp, Map);
    return;
  }

  if (CallBase *CB = dyn_cast<CallBase>(&Inst)) {
    maybeAddSuffixForNsanInterface(CB);
    if (CallInst *CI = dyn_cast<CallInst>(&Inst))
      maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);
    if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst)) {
      instrumentMemIntrinsic(MI);
      return;
    }
    populateShadowStack(*CB, TLI, Map);
    return;
  }

  if (ReturnInst *RetInst = dyn_cast<ReturnInst>(&Inst)) {
    if (!ClCheckRet)
      return;

    Value *RV = RetInst->getReturnValue();
    if (RV == nullptr)
      return; // This is a `ret void`.
    Type *const VT = RV->getType();
    Type *const ExtendedVT = Config.getExtendedFPType(VT);
    if (ExtendedVT == nullptr)
      return; // Not an FT ret.
    Value *RVShadow = Map.getShadow(RV);
    IRBuilder<> Builder(&Inst);
    Builder.SetCurrentDebugLocation(RetInst->getDebugLoc());

    RVShadow = emitCheck(RV, RVShadow, Builder, CheckLoc::makeRet());
    ++NumInstrumentedFTRets;
    // Store tag.
    Value *FnAddr =
        Builder.CreatePtrToInt(Inst.getParent()->getParent(), IntptrTy);
    Builder.CreateStore(FnAddr, NsanShadowRetTag);
    // Store value.
    Value *ShadowRetValPtr =
        Builder.CreateConstGEP2_64(NsanShadowRetType, NsanShadowRetPtr, 0, 0);
    Builder.CreateStore(RVShadow, ShadowRetValPtr);
    return;
  }

  if (InsertValueInst *Insert = dyn_cast<InsertValueInst>(&Inst)) {
    Value *V = Insert->getOperand(1);
    Type *const VT = V->getType();
    Type *const ExtendedVT = Config.getExtendedFPType(VT);
    if (ExtendedVT == nullptr)
      return;
    IRBuilder<> Builder(Insert);
    Builder.SetCurrentDebugLocation(Insert->getDebugLoc());
    emitCheck(V, Map.getShadow(V), Builder, CheckLoc::makeInsert());
    return;
  }
}

// Moves fast math flags from the function to individual instructions, and
// removes the attribute from the function.
// FIXME: Make this controllable with a flag.
static void moveFastMathFlags(Function &F,
                              std::vector<Instruction *> &Instructions) {
  FastMathFlags FMF;
#define MOVE_FLAG(attr, setter)                                                \
  if (F.getFnAttribute(attr).getValueAsString() == "true") {                   \
    F.removeFnAttr(attr);                                                      \
    FMF.set##setter();                                                         \
  }
  MOVE_FLAG("unsafe-fp-math", Fast)
  MOVE_FLAG("no-infs-fp-math", NoInfs)
  MOVE_FLAG("no-nans-fp-math", NoNaNs)
  MOVE_FLAG("no-signed-zeros-fp-math", NoSignedZeros)
#undef MOVE_FLAG

  for (Instruction *I : Instructions)
    if (isa<FPMathOperator>(I))
      I->setFastMathFlags(FMF);
}

bool NumericalStabilitySanitizer::sanitizeFunction(
    Function &F, const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __nsan_init from within
  // the module constructor.
  if (F.getName() == kNsanModuleCtorName)
    return false;
  if (!Config.initialize(&F.getParent()->getContext()))
    return false;
  initialize(*F.getParent());
  SmallVector<Instruction *, 8> AllLoadsAndStores;
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  if (!F.hasFnAttribute(Attribute::SanitizeNumericalStability))
    return false;

  // The instrumentation maintains:
  //  - for each IR value `v` of floating-point (or vector floating-point) type
  //    FT, a shadow IR value `s(v)` with twice the precision 2FT (e.g.
  //    double for float and f128 for double).
  //  - A shadow memory, which stores `s(v)` for any `v` that has been stored,
  //    along with a shadow memory tag, which stores whether the value in the
  //    corresponding shadow memory is valid. Note that this might be
  //    incorrect if a non-instrumented function stores to memory, or if
  //    memory is stored to through a char pointer.
  //  - A shadow stack, which holds `s(v)` for any floating-point argument `v`
  //    of a call to an instrumented function. This allows
  //    instrumented functions to retrieve the shadow values for their
  //    arguments.
  //    Because instrumented functions can be called from non-instrumented
  //    functions, the stack needs to include a tag so that the instrumented
  //    function knows whether shadow values are available for their
  //    parameters (i.e. whether is was called by an instrumented function).
  //    When shadow arguments are not available, they have to be recreated by
  //    extending the precision of the non-shadow arguments to the non-shadow
  //    value. Non-instrumented functions do not modify (or even know about) the
  //    shadow stack. The shadow stack pointer is __nsan_shadow_args. The shadow
  //    stack tag is __nsan_shadow_args_tag. The tag is any unique identifier
  //    for the function (we use the address of the function). Both variables
  //    are thread local.
  //    Example:
  //     calls                             shadow stack tag      shadow stack
  //     =======================================================================
  //     non_instrumented_1()              0                     0
  //             |
  //             v
  //     instrumented_2(float a)           0                     0
  //             |
  //             v
  //     instrumented_3(float b, double c) &instrumented_3       s(b),s(c)
  //             |
  //             v
  //     instrumented_4(float d)           &instrumented_4       s(d)
  //             |
  //             v
  //     non_instrumented_5(float e)       &non_instrumented_5   s(e)
  //             |
  //             v
  //     instrumented_6(float f)           &non_instrumented_5   s(e)
  //
  //   On entry, instrumented_2 checks whether the tag corresponds to its
  //   function ptr.
  //   Note that functions reset the tag to 0 after reading shadow parameters.
  //   This ensures that the function does not erroneously read invalid data if
  //   called twice in the same stack, once from an instrumented function and
  //   once from an uninstrumented one. For example, in the following example,
  //   resetting the tag in (A) ensures that (B) does not reuse the same the
  //   shadow arguments (which would be incorrect).
  //      instrumented_1(float a)
  //             |
  //             v
  //      instrumented_2(float b)  (A)
  //             |
  //             v
  //      non_instrumented_3()
  //             |
  //             v
  //      instrumented_2(float b)  (B)
  //
  //  - A shadow return slot. Any function that returns a floating-point value
  //    places a shadow return value in __nsan_shadow_ret_val. Again, because
  //    we might be calling non-instrumented functions, this value is guarded
  //    by __nsan_shadow_ret_tag marker indicating which instrumented function
  //    placed the value in __nsan_shadow_ret_val, so that the caller can check
  //    that this corresponds to the callee. Both variables are thread local.
  //
  //    For example, in the following example, the instrumentation in
  //    `instrumented_1` rejects the shadow return value from `instrumented_3`
  //    because is is not tagged as expected (`&instrumented_3` instead of
  //    `non_instrumented_2`):
  //
  //        instrumented_1()
  //            |
  //            v
  //        float non_instrumented_2()
  //            |
  //            v
  //        float instrumented_3()
  //
  // Calls of known math functions (sin, cos, exp, ...) are duplicated to call
  // their overload on the shadow type.

  // Collect all instructions before processing, as creating shadow values
  // creates new instructions inside the function.
  std::vector<Instruction *> OriginalInstructions;
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      OriginalInstructions.emplace_back(&Inst);
    }
  }

  moveFastMathFlags(F, OriginalInstructions);
  ValueToShadowMap ValueToShadow(&Config);

  // In the first pass, we create shadow values for all FT function arguments
  // and all phis. This ensures that the DFS of the next pass does not have
  // any loops.
  std::vector<PHINode *> OriginalPhis;
  createShadowArguments(F, TLI, ValueToShadow);
  for (Instruction *I : OriginalInstructions) {
    if (PHINode *Phi = dyn_cast<PHINode>(I)) {
      if (PHINode *Shadow = maybeCreateShadowPhi(*Phi, TLI)) {
        OriginalPhis.push_back(Phi);
        ValueToShadow.setShadow(Phi, Shadow);
      }
    }
  }

  // Create shadow values for all instructions creating FT values.
  for (Instruction *I : OriginalInstructions) {
    maybeCreateShadowValue(*I, TLI, ValueToShadow);
  }

  // Propagate shadow values across stores, calls and rets.
  for (Instruction *I : OriginalInstructions) {
    propagateShadowValues(*I, TLI, ValueToShadow);
  }

  // The last pass populates shadow phis with shadow values.
  for (PHINode *Phi : OriginalPhis) {
    PHINode *ShadowPhi = dyn_cast<PHINode>(ValueToShadow.getShadow(Phi));
    for (int I = 0, E = Phi->getNumOperands(); I < E; ++I) {
      Value *V = Phi->getOperand(I);
      Value *Shadow = ValueToShadow.getShadow(V);
      BasicBlock *IncomingBB = Phi->getIncomingBlock(I);
      // For some instructions (e.g. invoke), we create the shadow in a separate
      // block, different from the block where the original value is created.
      // In that case, the shadow phi might need to refer to this block instead
      // of the original block.
      // Note that this can only happen for instructions as constant shadows are
      // always created in the same block.
      ShadowPhi->addIncoming(Shadow, IncomingBB);
    }
  }

  return !ValueToShadow.empty();
}

// Instrument the memory intrinsics so that they properly modify the shadow
// memory.
bool NumericalStabilitySanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> Builder(MI);
  if (MemSetInst *M = dyn_cast<MemSetInst>(MI)) {
    Builder.SetCurrentDebugLocation(M->getDebugLoc());
    Builder.CreateCall(
        NsanSetValueUnknown,
        {// Address
         M->getArgOperand(0),
         // Size
         Builder.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(MI)) {
    Builder.SetCurrentDebugLocation(M->getDebugLoc());
    Builder.CreateCall(
        NsanCopyValues,
        {// Destination
         M->getArgOperand(0),
         // Source
         M->getArgOperand(1),
         // Size
         Builder.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
  }
  return false;
}

void NumericalStabilitySanitizer::maybeAddSuffixForNsanInterface(CallBase *CI) {
  Function *Fn = CI->getCalledFunction();
  if (Fn == nullptr)
    return;

  if (!Fn->getName().starts_with("__nsan_"))
    return;

  if (Fn->getName() == "__nsan_dump_shadow_mem") {
    assert(CI->arg_size() == 4 &&
           "invalid prototype for __nsan_dump_shadow_mem");
    // __nsan_dump_shadow_mem requires an extra parameter with the dynamic
    // configuration:
    // (shadow_type_id_for_long_double << 16) | (shadow_type_id_for_double << 8)
    // | shadow_type_id_for_double
    const uint64_t shadow_value_type_ids =
        (static_cast<size_t>(Config.byValueType(kLongDouble).getNsanTypeId())
         << 16) |
        (static_cast<size_t>(Config.byValueType(kDouble).getNsanTypeId())
         << 8) |
        static_cast<size_t>(Config.byValueType(kFloat).getNsanTypeId());
    CI->setArgOperand(3, ConstantInt::get(IntptrTy, shadow_value_type_ids));
  }
}
