//===-- AMDGPUCodeGenPrepare.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass does misc. AMDGPU optimizations on IR *just* before instruction
/// selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "amdgpu-late-codegenprepare"

using namespace llvm;

// Scalar load widening needs running after load-store-vectorizer as that pass
// doesn't handle overlapping cases. In addition, this pass enhances the
// widening to handle cases where scalar sub-dword loads are naturally aligned
// only but not dword aligned.
static cl::opt<bool>
    WidenLoads("amdgpu-late-codegenprepare-widen-constant-loads",
               cl::desc("Widen sub-dword constant address space loads in "
                        "AMDGPULateCodeGenPrepare"),
               cl::ReallyHidden, cl::init(true));

namespace {

class AMDGPULateCodeGenPrepare
    : public FunctionPass,
      public InstVisitor<AMDGPULateCodeGenPrepare, bool> {
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;

  AssumptionCache *AC = nullptr;
  UniformityInfo *UA = nullptr;

public:
  static char ID;

  AMDGPULateCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU IR late optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  bool visitInstruction(Instruction &) { return false; }

  // Check if the specified value is at least DWORD aligned.
  bool isDWORDAligned(const Value *V) const {
    KnownBits Known = computeKnownBits(V, *DL, 0, AC);
    return Known.countMinTrailingZeros() >= 2;
  }

  bool canWidenScalarExtLoad(LoadInst &LI) const;
  bool visitLoadInst(LoadInst &LI);
};

class LiveRegConversion {
private:
  // The instruction which defined the original virtual register used across
  // blocks
  Instruction *LiveRegDef;
  // The original type
  Type *OriginalType;
  // The desired type
  Type *NewType;
  // The instruction sequence that converts the virtual register, to be used
  // instead of the original
  Instruction *Converted = nullptr;
  // The builder used to build the conversion instruction
  IRBuilder<> ConvertBuilder;

public:
  // The instruction which defined the original virtual register used across
  // blocks
  Instruction *getLiveRegDef() { return LiveRegDef; }
  // The original type
  Type *getOriginalType() { return OriginalType; }
  // The desired type
  Type *getNewType() { return NewType; }
  void setNewType(Type *NewType) { this->NewType = NewType; }
  // The instruction that conerts the virtual register, to be used instead of
  // the original
  Instruction *getConverted() { return Converted; }
  void setConverted(Instruction *Converted) { this->Converted = Converted; }
  // The builder used to build the conversion instruction
  IRBuilder<> &getConvertBuilder() { return ConvertBuilder; }
  // Do we have a instruction sequence which convert the original virtual
  // register
  bool hasConverted() { return Converted != nullptr; }

  LiveRegConversion(Instruction *LiveRegDef, BasicBlock *InsertBlock,
                    BasicBlock::iterator InsertPt)
      : LiveRegDef(LiveRegDef), OriginalType(LiveRegDef->getType()),
        ConvertBuilder(InsertBlock, InsertPt) {}
  LiveRegConversion(Instruction *LiveRegDef, Type *NewType,
                    BasicBlock *InsertBlock, BasicBlock::iterator InsertPt)
      : LiveRegDef(LiveRegDef), OriginalType(LiveRegDef->getType()),
        NewType(NewType), ConvertBuilder(InsertBlock, InsertPt) {}
};

class LiveRegOptimizer {
private:
  Module *Mod = nullptr;
  // The scalar type to convert to
  Type *ConvertToScalar;
  // Holds the collection of PHIs with their pending new operands
  SmallVector<std::pair<Instruction *,
                        SmallVector<std::pair<Instruction *, BasicBlock *>, 4>>,
              4>
      PHIUpdater;

public:
  // Should the def of the instruction be converted if it is live across blocks
  bool shouldReplaceUses(const Instruction &I);
  // Convert the virtual register to the compatible vector of legal type
  void convertToOptType(LiveRegConversion &LR);
  // Convert the virtual register back to the original type, stripping away
  // the MSBs in cases where there was an imperfect fit (e.g. v2i32 -> v7i8)
  void convertFromOptType(LiveRegConversion &LR);
  // Get a vector of desired scalar type that is compatible with the original
  // vector. In cases where there is no bitsize equivalent using a legal vector
  // type, we pad the MSBs (e.g. v7i8 -> v2i32)
  Type *getCompatibleType(Instruction *InstToConvert);
  // Find and replace uses of the virtual register in different block with a
  // newly produced virtual register of legal type
  bool replaceUses(Instruction &I);
  // Replace the collected PHIs with newly produced incoming values. Replacement
  // is only done if we have a replacement for each original incoming value.
  bool replacePHIs();

  LiveRegOptimizer(Module *Mod) : Mod(Mod) {
    ConvertToScalar = Type::getInt32Ty(Mod->getContext());
  }
};

} // end anonymous namespace

bool AMDGPULateCodeGenPrepare::doInitialization(Module &M) {
  Mod = &M;
  DL = &Mod->getDataLayout();
  return false;
}

bool AMDGPULateCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  if (ST.hasScalarSubwordLoads())
    return false;

  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  UA = &getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();

  // "Optimize" the virtual regs that cross basic block boundaries. In such
  // cases, vectors of illegal types will be scalarized and widened, with each
  // scalar living in its own physical register. The optimization converts the
  // vectors to equivalent vectors of legal type (which are convereted back
  // before uses in subsequent blocks), to pack the bits into fewer physical
  // registers (used in CopyToReg/CopyFromReg pairs).
  LiveRegOptimizer LRO(Mod);

  bool Changed = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB)) {
      Changed |= visit(I);
      // GlobalISel should directly use the values, and do not need to emit
      // CopyTo/CopyFrom Regs across blocks
      if (TM.Options.EnableGlobalISel)
        continue;
      if (!LRO.shouldReplaceUses(I))
        continue;
      Changed |= LRO.replaceUses(I);
    }

  Changed |= LRO.replacePHIs();
  return Changed;
}

bool LiveRegOptimizer::replaceUses(Instruction &I) {
  bool MadeChange = false;

  struct ConvertUseInfo {
    Instruction *Converted;
    SmallVector<Instruction *, 4> Users;
  };
  DenseMap<BasicBlock *, ConvertUseInfo> UseConvertTracker;

  LiveRegConversion FromLRC(
      &I, I.getParent(),
      static_cast<BasicBlock::iterator>(std::next(I.getIterator())));
  FromLRC.setNewType(getCompatibleType(FromLRC.getLiveRegDef()));
  for (auto IUser = I.user_begin(); IUser != I.user_end(); IUser++) {

    if (Instruction *UserInst = dyn_cast<Instruction>(*IUser)) {
      if (UserInst->getParent() != I.getParent() || isa<PHINode>(UserInst)) {
        LLVM_DEBUG(dbgs() << *UserInst << "\n\tUses "
                          << *FromLRC.getOriginalType()
                          << " from previous block. Needs conversion\n");
        convertToOptType(FromLRC);
        if (!FromLRC.hasConverted())
          continue;
        // If it is a PHI node, just create and collect the new operand. We can
        // only replace the PHI node once we have converted all the operands
        if (auto PhiInst = dyn_cast<PHINode>(UserInst)) {
          for (unsigned Idx = 0; Idx < PhiInst->getNumIncomingValues(); Idx++) {
            Value *IncVal = PhiInst->getIncomingValue(Idx);
            if (&I == dyn_cast<Instruction>(IncVal)) {
              BasicBlock *IncBlock = PhiInst->getIncomingBlock(Idx);
              auto PHIOps = find_if(
                  PHIUpdater,
                  [&UserInst](
                      std::pair<Instruction *,
                                SmallVector<
                                    std::pair<Instruction *, BasicBlock *>, 4>>
                          &Entry) { return Entry.first == UserInst; });

              if (PHIOps == PHIUpdater.end())
                PHIUpdater.push_back(
                    {UserInst, {{FromLRC.getConverted(), IncBlock}}});
              else
                PHIOps->second.push_back({FromLRC.getConverted(), IncBlock});

              break;
            }
          }
          continue;
        }

        // Do not create multiple conversion sequences if there are multiple
        // uses in the same block
        if (UseConvertTracker.contains(UserInst->getParent())) {
          UseConvertTracker[UserInst->getParent()].Users.push_back(UserInst);
          LLVM_DEBUG(dbgs() << "\tUser already has access to converted def\n");
          continue;
        }

        LiveRegConversion ToLRC(FromLRC.getConverted(), I.getType(),
                                UserInst->getParent(),
                                static_cast<BasicBlock::iterator>(
                                    UserInst->getParent()->getFirstNonPHIIt()));
        convertFromOptType(ToLRC);
        assert(ToLRC.hasConverted());
        UseConvertTracker[UserInst->getParent()] = {ToLRC.getConverted(),
                                                    {UserInst}};
      }
    }
  }

  // Replace uses of with in a separate loop that is not dependent upon the
  // state of the uses
  for (auto &Entry : UseConvertTracker) {
    for (auto &UserInst : Entry.second.Users) {
      LLVM_DEBUG(dbgs() << *UserInst
                        << "\n\tNow uses: " << *Entry.second.Converted << "\n");
      UserInst->replaceUsesOfWith(&I, Entry.second.Converted);
      MadeChange = true;
    }
  }
  return MadeChange;
}

bool LiveRegOptimizer::replacePHIs() {
  bool MadeChange = false;
  for (auto Ele : PHIUpdater) {
    auto ThePHINode = cast<PHINode>(Ele.first);
    auto NewPHINodeOps = Ele.second;
    LLVM_DEBUG(dbgs() << "Attempting to replace: " << *ThePHINode << "\n");
    // If we have conveted all the required operands, then do the replacement
    if (ThePHINode->getNumIncomingValues() == NewPHINodeOps.size()) {
      IRBuilder<> Builder(Ele.first);
      auto NPHI = Builder.CreatePHI(NewPHINodeOps[0].first->getType(),
                                    NewPHINodeOps.size());
      for (auto IncVals : NewPHINodeOps) {
        NPHI->addIncoming(IncVals.first, IncVals.second);
        LLVM_DEBUG(dbgs() << "  Using: " << *IncVals.first
                          << "  For: " << IncVals.second->getName() << "\n");
      }
      LLVM_DEBUG(dbgs() << "Sucessfully replaced with " << *NPHI << "\n");
      LiveRegConversion ToLRC(NPHI, ThePHINode->getType(),
                              ThePHINode->getParent(),
                              static_cast<BasicBlock::iterator>(
                                  ThePHINode->getParent()->getFirstNonPHIIt()));
      convertFromOptType(ToLRC);
      assert(ToLRC.hasConverted());
      Ele.first->replaceAllUsesWith(ToLRC.getConverted());
      // The old PHI is no longer used
      ThePHINode->eraseFromParent();
      MadeChange = true;
    }
  }
  return MadeChange;
}

Type *LiveRegOptimizer::getCompatibleType(Instruction *InstToConvert) {
  Type *OriginalType = InstToConvert->getType();
  assert(OriginalType->getScalarSizeInBits() <=
         ConvertToScalar->getScalarSizeInBits());
  VectorType *VTy = dyn_cast<VectorType>(OriginalType);
  if (!VTy)
    return ConvertToScalar;

  unsigned OriginalSize =
      VTy->getScalarSizeInBits() * VTy->getElementCount().getFixedValue();
  unsigned ConvertScalarSize = ConvertToScalar->getScalarSizeInBits();
  unsigned ConvertEltCount =
      (OriginalSize + ConvertScalarSize - 1) / ConvertScalarSize;

  if (OriginalSize <= ConvertScalarSize)
    return IntegerType::get(Mod->getContext(), ConvertScalarSize);

  return VectorType::get(Type::getIntNTy(Mod->getContext(), ConvertScalarSize),
                         llvm::ElementCount::getFixed(ConvertEltCount));
}

void LiveRegOptimizer::convertToOptType(LiveRegConversion &LR) {
  if (LR.hasConverted()) {
    LLVM_DEBUG(dbgs() << "\tAlready has converted def\n");
    return;
  }

  VectorType *VTy = cast<VectorType>(LR.getOriginalType());

  Type *NewTy = LR.getNewType();
  assert(NewTy);
  VectorType *NewVTy = NewTy->isVectorTy() ? cast<VectorType>(NewTy) : nullptr;

  Value *V = static_cast<Value *>(LR.getLiveRegDef());
  unsigned OriginalSize =
      VTy->getScalarSizeInBits() * VTy->getElementCount().getFixedValue();
  unsigned NewSize = NewTy->isVectorTy()
                     ? NewVTy->getScalarSizeInBits() *
                           NewVTy->getElementCount().getFixedValue()
                     : NewTy->getScalarSizeInBits();

  auto &Builder = LR.getConvertBuilder();

  // If there is a bitsize match, we can fit the old vector into a new vector of
  // desired type
  if (OriginalSize == NewSize) {
    LR.setConverted(dyn_cast<Instruction>(Builder.CreateBitCast(V, NewTy)));
    LLVM_DEBUG(dbgs() << "\tConverted def to " << *LR.getConverted()->getType()
                      << "\n");
    return;
  }

  // If there is a bitsize mismatch, we must use a wider vector
  assert(NewSize > OriginalSize);
  ElementCount ExpandedVecElementCount =
      llvm::ElementCount::getFixed(NewSize / VTy->getScalarSizeInBits());

  SmallVector<int, 8> ShuffleMask;
  for (unsigned I = 0; I < VTy->getElementCount().getFixedValue(); I++)
    ShuffleMask.push_back(I);

  for (uint64_t I = VTy->getElementCount().getFixedValue();
       I < ExpandedVecElementCount.getFixedValue(); I++)
    ShuffleMask.push_back(VTy->getElementCount().getFixedValue());

  auto ExpandedVec =
      dyn_cast<Instruction>(Builder.CreateShuffleVector(V, ShuffleMask));
  LR.setConverted(
      dyn_cast<Instruction>(Builder.CreateBitCast(ExpandedVec, NewTy)));
  LLVM_DEBUG(dbgs() << "\tConverted def to " << *LR.getConverted()->getType()
                    << "\n");
  return;
}

void LiveRegOptimizer::convertFromOptType(LiveRegConversion &LRC) {
  Type *OTy = LRC.getOriginalType();
  VectorType *VTy =
      OTy->isVectorTy() ? dyn_cast<VectorType>(LRC.getOriginalType()) : nullptr;

  VectorType *NewVTy = cast<VectorType>(LRC.getNewType());

  Value *V = static_cast<Value *>(LRC.getLiveRegDef());
  unsigned OriginalSize =
      OTy->isVectorTy()
          ? VTy->getScalarSizeInBits() * VTy->getElementCount().getFixedValue()
          : OTy->getScalarSizeInBits();
  unsigned NewSize =
      NewVTy->getScalarSizeInBits() * NewVTy->getElementCount().getFixedValue();

  auto &Builder = LRC.getConvertBuilder();

  // If there is a bitsize match, we simply convert back to the original type
  if (OriginalSize == NewSize) {
    LRC.setConverted(dyn_cast<Instruction>(Builder.CreateBitCast(V, NewVTy)));
    LLVM_DEBUG(dbgs() << "\tProduced for user: " << *LRC.getConverted()
                      << "\n");
    return;
  }

  if (!OTy->isVectorTy()) {
    auto Trunc = dyn_cast<Instruction>(Builder.CreateTrunc(
        LRC.getLiveRegDef(), IntegerType::get(Mod->getContext(), NewSize)));
    auto Original = dyn_cast<Instruction>(Builder.CreateBitCast(Trunc, NewVTy));
    LRC.setConverted(dyn_cast<Instruction>(Original));
    LLVM_DEBUG(dbgs() << "\tProduced for user: " << *LRC.getConverted()
                      << "\n");
    return;
  }

  // If there is a bitsize mismatch, we have used a wider vector and must strip
  // the MSBs to convert back to the original type
  assert(OriginalSize > NewSize);
  ElementCount ExpandedVecElementCount = llvm::ElementCount::getFixed(
      OriginalSize / NewVTy->getScalarSizeInBits());
  VectorType *ExpandedVT = VectorType::get(
      Type::getIntNTy(Mod->getContext(), NewVTy->getScalarSizeInBits()),
      ExpandedVecElementCount);
  Instruction *Converted = dyn_cast<Instruction>(
      Builder.CreateBitCast(LRC.getLiveRegDef(), ExpandedVT));

  unsigned NarrowElementCount = NewVTy->getElementCount().getFixedValue();
  SmallVector<int, 8> ShuffleMask;
  for (uint64_t I = 0; I < NarrowElementCount; I++)
    ShuffleMask.push_back(I);

  Instruction *NarrowVec = dyn_cast<Instruction>(
      Builder.CreateShuffleVector(Converted, ShuffleMask));
  LRC.setConverted(dyn_cast<Instruction>(NarrowVec));
  LLVM_DEBUG(dbgs() << "\tProduced for user: " << *LRC.getConverted() << "\n");
  return;
}

bool LiveRegOptimizer::shouldReplaceUses(const Instruction &I) {
  // Vectors of illegal types are copied across blocks in an efficient manner.
  // They are scalarized and widened to legal scalars. In such cases, we can do
  // better by using legal vector types
  Type *IType = I.getType();
  return IType->isVectorTy() && IType->getScalarSizeInBits() < 16 &&
         !I.getType()->getScalarType()->isPointerTy();
}

bool AMDGPULateCodeGenPrepare::canWidenScalarExtLoad(LoadInst &LI) const {
  unsigned AS = LI.getPointerAddressSpace();
  // Skip non-constant address space.
  if (AS != AMDGPUAS::CONSTANT_ADDRESS &&
      AS != AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return false;
  // Skip non-simple loads.
  if (!LI.isSimple())
    return false;
  Type *Ty = LI.getType();
  // Skip aggregate types.
  if (Ty->isAggregateType())
    return false;
  unsigned TySize = DL->getTypeStoreSize(Ty);
  // Only handle sub-DWORD loads.
  if (TySize >= 4)
    return false;
  // That load must be at least naturally aligned.
  if (LI.getAlign() < DL->getABITypeAlign(Ty))
    return false;
  // It should be uniform, i.e. a scalar load.
  return UA->isUniform(&LI);
}

bool AMDGPULateCodeGenPrepare::visitLoadInst(LoadInst &LI) {
  if (!WidenLoads)
    return false;

  // Skip if that load is already aligned on DWORD at least as it's handled in
  // SDAG.
  if (LI.getAlign() >= 4)
    return false;

  if (!canWidenScalarExtLoad(LI))
    return false;

  int64_t Offset = 0;
  auto *Base =
      GetPointerBaseWithConstantOffset(LI.getPointerOperand(), Offset, *DL);
  // If that base is not DWORD aligned, it's not safe to perform the following
  // transforms.
  if (!isDWORDAligned(Base))
    return false;

  int64_t Adjust = Offset & 0x3;
  if (Adjust == 0) {
    // With a zero adjust, the original alignment could be promoted with a
    // better one.
    LI.setAlignment(Align(4));
    return true;
  }

  IRBuilder<> IRB(&LI);
  IRB.SetCurrentDebugLocation(LI.getDebugLoc());

  unsigned LdBits = DL->getTypeStoreSizeInBits(LI.getType());
  auto IntNTy = Type::getIntNTy(LI.getContext(), LdBits);

  auto *NewPtr = IRB.CreateConstGEP1_64(
      IRB.getInt8Ty(),
      IRB.CreateAddrSpaceCast(Base, LI.getPointerOperand()->getType()),
      Offset - Adjust);

  LoadInst *NewLd = IRB.CreateAlignedLoad(IRB.getInt32Ty(), NewPtr, Align(4));
  NewLd->copyMetadata(LI);
  NewLd->setMetadata(LLVMContext::MD_range, nullptr);

  unsigned ShAmt = Adjust * 8;
  auto *NewVal = IRB.CreateBitCast(
      IRB.CreateTrunc(IRB.CreateLShr(NewLd, ShAmt), IntNTy), LI.getType());
  LI.replaceAllUsesWith(NewVal);
  RecursivelyDeleteTriviallyDeadInstructions(&LI);

  return true;
}

INITIALIZE_PASS_BEGIN(AMDGPULateCodeGenPrepare, DEBUG_TYPE,
                      "AMDGPU IR late optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPULateCodeGenPrepare, DEBUG_TYPE,
                    "AMDGPU IR late optimizations", false, false)

char AMDGPULateCodeGenPrepare::ID = 0;

FunctionPass *llvm::createAMDGPULateCodeGenPreparePass() {
  return new AMDGPULateCodeGenPrepare();
}
