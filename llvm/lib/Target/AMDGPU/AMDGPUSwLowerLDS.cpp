//===-- AMDGPUSwLowerLDS.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the local data store, LDS, uses in kernel and non-kernel
// functions in module with dynamically allocated device global memory.
//
// Replacement of Kernel LDS accesses:
//    For a kernel, LDS access can be static or dynamic which are direct
//    (accessed within kernel) and indirect (accessed through non-kernels).
//    A device global memory equal to size of all these LDS globals will be
//    allocated. At the prologue of the kernel, a single work-item from the
//    work-group, does a "malloc" and stores the pointer of the allocation in
//    new LDS global that will be created for the kernel. This will be called
//    "malloc LDS global" in this pass.
//    Each LDS access corresponds to an offset in the allocated memory.
//    All static LDS accesses will be allocated first and then dynamic LDS
//    will occupy the device global memoery.
//    To store the offsets corresponding to all LDS accesses, another global
//    variable is created which will be called "metadata global" in this pass.
//    - Malloc LDS Global:
//        It is LDS global of ptr type with name
//        "llvm.amdgcn.sw.lds.<kernel-name>".
//    - Metadata Global:
//        It is of struct type, with n members. n equals the number of LDS
//        globals accessed by the kernel(direct and indirect). Each member of
//        struct is another struct of type {i32, i32}. First member corresponds
//        to offset, second member corresponds to size of LDS global being
//        replaced. It will have name "llvm.amdgcn.sw.lds.<kernel-name>.md".
//        This global will have an intializer with static LDS related offsets
//        and sizes initialized. But for dynamic LDS related entries, offsets
//        will be intialized to previous static LDS allocation end offset. Sizes
//        for them will be zero initially. These dynamic LDS offset and size
//        values will be updated with in the kernel, since kernel can read the
//        dynamic LDS size allocation done at runtime with query to
//        "hidden_dynamic_lds_size" hidden kernel argument.
//
//    LDS accesses within the kernel will be replaced by "gep" ptr to
//    corresponding offset into allocated device global memory for the kernel.
//    At the epilogue of kernel, allocated memory would be made free by the same
//    single work-item.
//
// Replacement of non-kernel LDS accesses:
//    Multiple kernels can access the same non-kernel function.
//    All the kernels accessing LDS through non-kernels are sorted and
//    assigned a kernel-id. All the LDS globals accessed by non-kernels
//    are sorted. This information is used to build two tables:
//    - Base table:
//        Base table will have single row, with elements of the row
//        placed as per kernel ID. Each element in the row corresponds
//        to addresss of "malloc LDS global" variable created for
//        that kernel.
//    - Offset table:
//        Offset table will have multiple rows and columns.
//        Rows are assumed to be from 0 to (n-1). n is total number
//        of kernels accessing the LDS through non-kernels.
//        Each row will have m elements. m is the total number of
//        unique LDS globals accessed by all non-kernels.
//        Each element in the row correspond to the address of
//        the replacement of LDS global done by that particular kernel.
//    A LDS variable in non-kernel will be replaced based on the information
//    from base and offset tables. Based on kernel-id query, address of "malloc
//    LDS global" for that corresponding kernel is obtained from base table.
//    The Offset into the base "malloc LDS global" is obtained from
//    corresponding element in offset table. With this information, replacement
//    value is obtained.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>

#define DEBUG_TYPE "amdgpu-sw-lower-lds"

using namespace llvm;
using namespace AMDGPU;

namespace {

using DomTreeCallback = function_ref<DominatorTree *(Function &F)>;

struct LDSAccessTypeInfo {
  SetVector<GlobalVariable *> StaticLDSGlobals;
  SetVector<GlobalVariable *> DynamicLDSGlobals;
};

// Struct to hold all the Metadata required for a kernel
// to replace a LDS global uses with corresponding offset
// in to device global memory.
struct KernelLDSParameters {
  GlobalVariable *MallocLDSGlobal{nullptr};
  GlobalVariable *MallocMetadataGlobal{nullptr};
  LDSAccessTypeInfo DirectAccess;
  LDSAccessTypeInfo IndirectAccess;
  DenseMap<GlobalVariable *, SmallVector<uint32_t, 3>>
      LDSToReplacementIndicesMap;
  int32_t KernelId{-1};
  uint32_t MallocSize{0};
};

// Struct to store infor for creation of offset table
// for all the non-kernel LDS accesses.
struct NonKernelLDSParameters {
  GlobalVariable *LDSBaseTable{nullptr};
  GlobalVariable *LDSOffsetTable{nullptr};
  SetVector<Function *> OrderedKernels;
  SetVector<GlobalVariable *> OrdereLDSGlobals;
};

class AMDGPUSwLowerLDS {
public:
  AMDGPUSwLowerLDS(Module &mod, DomTreeCallback Callback)
      : M(mod), IRB(M.getContext()), DTCallback(Callback) {}
  bool Run();
  void GetUsesOfLDSByNonKernels(CallGraph const &CG,
                                FunctionVariableMap &functions);
  SetVector<Function *>
  GetOrderedIndirectLDSAccessingKernels(SetVector<Function *> &&Kernels);
  SetVector<GlobalVariable *>
  GetOrderedNonKernelAllLDSGlobals(SetVector<GlobalVariable *> &&Variables);
  void PopulateMallocLDSGlobal(Function *Func);
  void PopulateMallocMetadataGlobal(Function *Func);
  void PopulateLDSToReplacementIndicesMap(Function *Func);
  void ReplaceKernelLDSAccesses(Function *Func);
  void LowerKernelLDSAccesses(Function *Func, DomTreeUpdater &DTU);
  void BuildNonKernelLDSOffsetTable(
      std::shared_ptr<NonKernelLDSParameters> &NKLDSParams);
  void BuildNonKernelLDSBaseTable(
      std::shared_ptr<NonKernelLDSParameters> &NKLDSParams);
  Constant *
  GetAddressesOfVariablesInKernel(Function *Func,
                                  SetVector<GlobalVariable *> &Variables);
  void LowerNonKernelLDSAccesses(
      Function *Func, SetVector<GlobalVariable *> &LDSGlobals,
      std::shared_ptr<NonKernelLDSParameters> &NKLDSParams);

private:
  Module &M;
  IRBuilder<> IRB;
  DomTreeCallback DTCallback;
  DenseMap<Function *, std::shared_ptr<KernelLDSParameters>>
      KernelToLDSParametersMap;
};

template <typename T> SetVector<T> SortByName(std::vector<T> &&V) {
  // Sort the vector of globals or Functions based on their name.
  // Returns a SetVector of globals/Functions.
  llvm::sort(V.begin(), V.end(), [](const auto *L, const auto *R) {
    return L->getName() < R->getName();
  });
  return {std::move(SetVector<T>(V.begin(), V.end()))};
}

SetVector<GlobalVariable *> AMDGPUSwLowerLDS::GetOrderedNonKernelAllLDSGlobals(
    SetVector<GlobalVariable *> &&Variables) {
  // Sort all the non-kernel LDS accesses based on theor name.
  SetVector<GlobalVariable *> Ordered = SortByName(
      std::vector<GlobalVariable *>(Variables.begin(), Variables.end()));
  return std::move(Ordered);
}

SetVector<Function *> AMDGPUSwLowerLDS::GetOrderedIndirectLDSAccessingKernels(
    SetVector<Function *> &&Kernels) {
  // Sort the non-kernels accessing LDS based on theor name.
  // Also assign a kernel ID metadata based on the sorted order.
  LLVMContext &Ctx = M.getContext();
  if (Kernels.size() > UINT32_MAX) {
    // 32 bit keeps it in one SGPR. > 2**32 kernels won't fit on the GPU
    report_fatal_error("Unimplemented SW LDS lowering for > 2**32 kernels");
  }
  SetVector<Function *> OrderedKernels =
      SortByName(std::vector<Function *>(Kernels.begin(), Kernels.end()));
  for (size_t i = 0; i < Kernels.size(); i++) {
    Metadata *AttrMDArgs[1] = {
        ConstantAsMetadata::get(IRB.getInt32(i)),
    };
    Function *Func = OrderedKernels[i];
    Func->setMetadata("llvm.amdgcn.lds.kernel.id",
                      MDNode::get(Ctx, AttrMDArgs));
    auto &LDSParams = KernelToLDSParametersMap[Func];
    assert(LDSParams);
    LDSParams->KernelId = i;
  }
  return std::move(OrderedKernels);
}

void AMDGPUSwLowerLDS::GetUsesOfLDSByNonKernels(
    CallGraph const &CG, FunctionVariableMap &functions) {
  // Get uses from the current function, excluding uses by called functions
  // Two output variables to avoid walking the globals list twice
  for (auto &GV : M.globals()) {
    if (!AMDGPU::isLDSVariableToLower(GV)) {
      continue;
    }

    if (GV.isAbsoluteSymbolRef()) {
      report_fatal_error(
          "LDS variables with absolute addresses are unimplemented.");
    }

    for (User *V : GV.users()) {
      User *FUU = V;
      bool isCast = isa<BitCastOperator, AddrSpaceCastOperator>(FUU);
      if (isCast && FUU->hasOneUse() && !FUU->user_begin()->user_empty())
        FUU = *FUU->user_begin();
      if (auto *I = dyn_cast<Instruction>(FUU)) {
        Function *F = I->getFunction();
        if (!isKernelLDS(F)) {
          functions[F].insert(&GV);
        }
      }
    }
  }
}

void AMDGPUSwLowerLDS::PopulateMallocLDSGlobal(Function *Func) {
  // Create new LDS global required for each kernel to store
  // device global memory pointer.
  auto &LDSParams = KernelToLDSParametersMap[Func];
  assert(LDSParams);
  // create new global pointer variable
  LDSParams->MallocLDSGlobal = new GlobalVariable(
      M, IRB.getPtrTy(), false, GlobalValue::InternalLinkage,
      PoisonValue::get(IRB.getPtrTy()),
      Twine("llvm.amdgcn.sw.lds." + Func->getName()), nullptr,
      GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
  return;
}

void AMDGPUSwLowerLDS::PopulateMallocMetadataGlobal(Function *Func) {
  // Create new metadata global for every kernel and initialize the
  // start offsets and sizes corresponding to each LDS accesses.
  auto &LDSParams = KernelToLDSParametersMap[Func];
  assert(LDSParams);
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  std::vector<Type *> Items;
  Type *Int32Ty = IRB.getInt32Ty();
  std::vector<Constant *> Initializers;
  Align MaxAlignment(1);
  auto UpdateMaxAlignment = [&MaxAlignment, &DL](GlobalVariable *GV) {
    uint32_t GVAlignValue = GV->getAlignment();
    Align GVAlign =
        GVAlignValue ? Align(GVAlignValue) : AMDGPU::getAlign(DL, GV);
    MaxAlignment = std::max(MaxAlignment, GVAlign);
  };

  for (GlobalVariable *GV : LDSParams->DirectAccess.StaticLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams->DirectAccess.DynamicLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams->IndirectAccess.StaticLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams->IndirectAccess.DynamicLDSGlobals)
    UpdateMaxAlignment(GV);

  uint32_t MaxAlignValue = MaxAlignment.value();

  //{StartOffset, SizeInBytes}
  StructType *LDSItemTy = StructType::create(
      Ctx, {Int32Ty, Int32Ty},
      "llvm.amdgcn.sw.lds." + Func->getName().str() + ".md.item");

  auto InitializerLamda = [&](SetVector<GlobalVariable *> &LDSGlobals) {
    for (auto &GV : LDSGlobals) {
      Type *Ty = GV->getValueType();
      const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
      Items.push_back(LDSItemTy);
      Constant *ItemStartOffset =
          ConstantInt::get(Int32Ty, LDSParams->MallocSize);
      Constant *SizeInBytesConst = ConstantInt::get(Int32Ty, SizeInBytes);
      uint64_t AlignedSize =
          ((SizeInBytes + MaxAlignValue - 1) / MaxAlignValue) * MaxAlignValue;
      LDSParams->MallocSize += AlignedSize;
      Constant *InitItem =
          ConstantStruct::get(LDSItemTy, {ItemStartOffset, SizeInBytesConst});
      Initializers.push_back(InitItem);
    }
  };

  InitializerLamda(LDSParams->DirectAccess.StaticLDSGlobals);
  InitializerLamda(LDSParams->IndirectAccess.StaticLDSGlobals);
  InitializerLamda(LDSParams->DirectAccess.DynamicLDSGlobals);
  InitializerLamda(LDSParams->IndirectAccess.DynamicLDSGlobals);

  StructType *MetadataStructType = StructType::create(
      Ctx, Items, ("llvm.amdgcn.sw.lds." + Func->getName().str() + ".md.type"));
  LDSParams->MallocMetadataGlobal = new GlobalVariable(
      M, MetadataStructType, false, GlobalValue::InternalLinkage,
      PoisonValue::get(MetadataStructType),
      ("llvm.amdgcn.sw.lds." + Func->getName().str() + ".md"), nullptr,
      GlobalValue::NotThreadLocal, AMDGPUAS::GLOBAL_ADDRESS, false);
  Constant *data = ConstantStruct::get(MetadataStructType, Initializers);
  LDSParams->MallocMetadataGlobal->setInitializer(data);
  LDSParams->MallocMetadataGlobal->setAlignment(MaxAlignment);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  LDSParams->MallocMetadataGlobal->setSanitizerMetadata(MD);
  return;
}

void AMDGPUSwLowerLDS::PopulateLDSToReplacementIndicesMap(Function *Func) {
  // Fill the corresponding LDS replacement indices for each LDS access
  // related to this kernel.
  auto &LDSParams = KernelToLDSParametersMap[Func];
  assert(LDSParams);
  auto PopulateIndices = [&](SetVector<GlobalVariable *> &LDSGlobals,
                             uint32_t &Idx) {
    for (auto &GV : LDSGlobals) {
      LDSParams->LDSToReplacementIndicesMap[GV] = {0, Idx, 0};
      ++Idx;
    }
  };
  uint32_t Idx = 0;
  PopulateIndices(LDSParams->DirectAccess.StaticLDSGlobals, Idx);
  PopulateIndices(LDSParams->IndirectAccess.StaticLDSGlobals, Idx);
  PopulateIndices(LDSParams->DirectAccess.DynamicLDSGlobals, Idx);
  PopulateIndices(LDSParams->IndirectAccess.DynamicLDSGlobals, Idx);
  return;
}

static void ReplacesUsesOfGlobalInFunction(Function *Func, GlobalVariable *GV,
                                           Value *Replacement) {
  // Replace all uses of LDS global in this Function with a Replacement.
  auto ReplaceUsesLambda = [Func](const Use &U) -> bool {
    auto *FUU = U.getUser();
    bool isCast = isa<BitCastOperator, AddrSpaceCastOperator>(FUU);
    if (isCast && FUU->hasOneUse() && !FUU->user_begin()->user_empty())
      FUU = *FUU->user_begin();
    if (auto *inst = llvm::dyn_cast<Instruction>(FUU)) {
      auto *Func1 = inst->getParent()->getParent();
      if (Func == Func1)
        return true;
    }
    return false;
  };
  GV->replaceUsesWithIf(Replacement, ReplaceUsesLambda);
  return;
}

void AMDGPUSwLowerLDS::ReplaceKernelLDSAccesses(Function *Func) {
  auto &LDSParams = KernelToLDSParametersMap[Func];
  assert(LDSParams);
  GlobalVariable *MallocLDSGlobal = LDSParams->MallocLDSGlobal;
  assert(MallocLDSGlobal);
  GlobalVariable *MallocMetadataGlobal = LDSParams->MallocMetadataGlobal;
  assert(MallocMetadataGlobal);
  StructType *MallocMetadataStructType =
      cast<StructType>(MallocMetadataGlobal->getValueType());
  Type *Int32Ty = IRB.getInt32Ty();

  // Replace all uses of LDS global in this Function with a Replacement.
  auto ReplaceLDSGlobalUses = [&](SetVector<GlobalVariable *> &LDSGlobals) {
    for (auto &GV : LDSGlobals) {
      // Do not generate instructions if LDS access is in non-kernel
      // i.e indirect-access.
      if ((LDSParams->IndirectAccess.StaticLDSGlobals.contains(GV) ||
           LDSParams->IndirectAccess.DynamicLDSGlobals.contains(GV)) &&
          (!LDSParams->DirectAccess.StaticLDSGlobals.contains(GV) &&
           !LDSParams->DirectAccess.DynamicLDSGlobals.contains(GV)))
        continue;
      auto &Indices = LDSParams->LDSToReplacementIndicesMap[GV];
      assert(Indices.size() == 3);
      uint32_t Idx0 = Indices[0];
      uint32_t Idx1 = Indices[1];
      uint32_t Idx2 = Indices[2];
      Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, Idx0),
                            ConstantInt::get(Int32Ty, Idx1),
                            ConstantInt::get(Int32Ty, Idx2)};
      Constant *GEP = ConstantExpr::getGetElementPtr(
          MallocMetadataStructType, MallocMetadataGlobal, GEPIdx, true);
      Value *Load = IRB.CreateLoad(Int32Ty, GEP);
      Value *BasePlusOffset =
          IRB.CreateInBoundsGEP(GV->getType(), MallocLDSGlobal, {Load});
      ReplacesUsesOfGlobalInFunction(Func, GV, BasePlusOffset);
    }
  };
  ReplaceLDSGlobalUses(LDSParams->DirectAccess.StaticLDSGlobals);
  ReplaceLDSGlobalUses(LDSParams->IndirectAccess.StaticLDSGlobals);
  ReplaceLDSGlobalUses(LDSParams->DirectAccess.DynamicLDSGlobals);
  ReplaceLDSGlobalUses(LDSParams->IndirectAccess.DynamicLDSGlobals);
  return;
}

void AMDGPUSwLowerLDS::LowerKernelLDSAccesses(Function *Func,
                                              DomTreeUpdater &DTU) {
  auto &LDSParams = KernelToLDSParametersMap[Func];
  assert(LDSParams);
  auto &Ctx = M.getContext();
  auto *PrevEntryBlock = &Func->getEntryBlock();

  // Create malloc block.
  auto *MallocBlock = BasicBlock::Create(Ctx, "Malloc", Func, PrevEntryBlock);

  // Create WIdBlock block which has instructions related to selection of
  // {0,0,0} indiex work item in the work group.
  auto *WIdBlock = BasicBlock::Create(Ctx, "WId", Func, MallocBlock);
  IRB.SetInsertPoint(WIdBlock, WIdBlock->begin());
  auto *const WIdx =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_x, {}, {});
  auto *const WIdy =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_y, {}, {});
  auto *const WIdz =
      IRB.CreateIntrinsic(Intrinsic::amdgcn_workitem_id_z, {}, {});
  auto *const XYOr = IRB.CreateOr(WIdx, WIdy);
  auto *const XYZOr = IRB.CreateOr(XYOr, WIdz);
  auto *const WIdzCond = IRB.CreateICmpEQ(XYZOr, IRB.getInt32(0));

  GlobalVariable *MallocMetadataGlobal = LDSParams->MallocMetadataGlobal;
  assert(MallocMetadataGlobal);
  StructType *MetadataStructType =
      cast<StructType>(MallocMetadataGlobal->getValueType());

  // All work items will branch to PrevEntryBlock except {0,0,0} index
  // work item which will branch to malloc block.
  IRB.CreateCondBr(WIdzCond, MallocBlock, PrevEntryBlock);

  // Malloc block
  IRB.SetInsertPoint(MallocBlock, MallocBlock->begin());

  // If Dynamic LDS globals are accessed by the kernel,
  // Get the size of dyn lds from hidden dyn_lds_size kernel arg.
  // Update the corresponding metadata global entries for this dyn lds global.
  uint32_t MallocSize = LDSParams->MallocSize;
  Value *CurrMallocSize = IRB.getInt64(MallocSize);
  if (!LDSParams->DirectAccess.DynamicLDSGlobals.empty() ||
      !LDSParams->IndirectAccess.DynamicLDSGlobals.empty()) {
    unsigned MaxAlignment = MallocMetadataGlobal->getAlignment();
    Value *MaxAlignValue = IRB.getInt64(MaxAlignment);
    Value *MaxAlignValueMinusOne = IRB.getInt64(MaxAlignment - 1);
    auto MallocSizeCalcLambda =
        [&](SetVector<GlobalVariable *> &DynamicLDSGlobals) {
          for (GlobalVariable *DynGV : DynamicLDSGlobals) {
            auto &Indices = LDSParams->LDSToReplacementIndicesMap[DynGV];
            // Get size from hidden dyn_lds_size argument of kernel int
            // CurrDynLDSSize
            Value *ImplicitArg =
                IRB.CreateIntrinsic(Intrinsic::amdgcn_implicitarg_ptr, {}, {});
            Value *HiddenDynLDSSize = IRB.CreateInBoundsGEP(
                ImplicitArg->getType(), ImplicitArg, {IRB.getInt32(15)});
            Value *CurrDynLDSSize =
                IRB.CreateLoad(IRB.getInt64Ty(), HiddenDynLDSSize);
            auto *GEPForOffset = IRB.CreateInBoundsGEP(
                MetadataStructType, MallocMetadataGlobal,
                {IRB.getInt32(0), IRB.getInt32(Indices[1]), IRB.getInt32(0)});
            IRB.CreateStore(CurrMallocSize, GEPForOffset);

            auto *GEPForSize = IRB.CreateInBoundsGEP(
                MetadataStructType, MallocMetadataGlobal,
                {IRB.getInt32(0), IRB.getInt32(Indices[1]), IRB.getInt32(1)});
            IRB.CreateStore(CurrDynLDSSize, GEPForSize);
            CurrMallocSize = IRB.CreateAdd(CurrMallocSize, CurrDynLDSSize);
            CurrMallocSize =
                IRB.CreateAdd(CurrMallocSize, MaxAlignValueMinusOne);
            CurrMallocSize = IRB.CreateUDiv(CurrMallocSize, MaxAlignValue);
            CurrMallocSize = IRB.CreateMul(CurrMallocSize, MaxAlignValue);
          }
        };
    MallocSizeCalcLambda(LDSParams->DirectAccess.DynamicLDSGlobals);
    MallocSizeCalcLambda(LDSParams->IndirectAccess.DynamicLDSGlobals);
  }

  // Create a call to malloc function which does device global memory allocation
  // with size equals to all LDS global accesses size  in this kernel.
  const char MallocImplName[] = "malloc";
  FunctionCallee AMDGPUMallocReturn = M.getOrInsertFunction(
      MallocImplName,
      FunctionType::get(IRB.getPtrTy(1), {IRB.getInt64Ty()}, false));
  Value *MCI = IRB.CreateCall(AMDGPUMallocReturn, {CurrMallocSize});

  GlobalVariable *MallocLDSGlobal = LDSParams->MallocLDSGlobal;
  assert(MallocLDSGlobal);

  // create load of malloc to new global
  IRB.CreateStore(MCI, MallocLDSGlobal);

  // Create branch to PrevEntryBlock
  IRB.CreateBr(PrevEntryBlock);

  // Create wave-group barrier at the starting of Previous entry block
  Type *Int1Ty = IRB.getInt1Ty();
  IRB.SetInsertPoint(PrevEntryBlock, PrevEntryBlock->begin());
  auto *XYZCondPhi = IRB.CreatePHI(Int1Ty, 2, "xyzCond");
  XYZCondPhi->addIncoming(IRB.getInt1(0), WIdBlock);
  XYZCondPhi->addIncoming(IRB.getInt1(1), MallocBlock);

  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});

  ReplaceKernelLDSAccesses(Func);

  auto *CondFreeBlock = BasicBlock::Create(Ctx, "CondFree", Func);
  auto *FreeBlock = BasicBlock::Create(Ctx, "Free", Func);
  auto *EndBlock = BasicBlock::Create(Ctx, "End", Func);
  for (BasicBlock &BB : *Func) {
    if (!BB.empty()) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&BB.back())) {
        BasicBlock *Block = &BB;
        RI->eraseFromParent();
        IRB.SetInsertPoint(&BB, BB.end());
        IRB.CreateBr(CondFreeBlock);
      }
    }
  }

  // Cond Free Block
  IRB.SetInsertPoint(CondFreeBlock, CondFreeBlock->begin());
  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});
  IRB.CreateCondBr(XYZCondPhi, FreeBlock, EndBlock);

  // Free Block
  IRB.SetInsertPoint(FreeBlock, FreeBlock->begin());

  // Free the previously allocate device global memory.
  const char FreeImplName[] = "free";
  FunctionCallee AMDGPUFreeReturn = M.getOrInsertFunction(
      FreeImplName,
      FunctionType::get(IRB.getVoidTy(), {IRB.getPtrTy()}, false));

  Value *MallocPtr = IRB.CreateLoad(IRB.getPtrTy(), MallocLDSGlobal);
  IRB.CreateCall(AMDGPUFreeReturn, {MallocPtr});
  IRB.CreateBr(EndBlock);

  // End Block
  IRB.SetInsertPoint(EndBlock, EndBlock->begin());
  IRB.CreateRetVoid();
  // Update the DomTree with corresponding links to basic blocks.
  DTU.applyUpdates({{DominatorTree::Insert, WIdBlock, MallocBlock},
                    {DominatorTree::Insert, MallocBlock, PrevEntryBlock},
                    {DominatorTree::Insert, CondFreeBlock, FreeBlock},
                    {DominatorTree::Insert, FreeBlock, EndBlock}});
  return;
}

Constant *AMDGPUSwLowerLDS::GetAddressesOfVariablesInKernel(
    Function *Func, SetVector<GlobalVariable *> &Variables) {
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  auto &LDSParams = KernelToLDSParametersMap[Func];
  assert(LDSParams);

  GlobalVariable *MallocMetadataGlobal = LDSParams->MallocMetadataGlobal;
  assert(MallocMetadataGlobal);
  StructType *MallocMetadataStructType =
      cast<StructType>(MallocMetadataGlobal->getValueType());
  ArrayType *KernelOffsetsType = ArrayType::get(Int32Ty, Variables.size());

  SmallVector<Constant *> Elements;
  for (size_t i = 0; i < Variables.size(); i++) {
    GlobalVariable *GV = Variables[i];
    assert(GV);
    if (LDSParams->LDSToReplacementIndicesMap.contains(GV)) {
      auto &Indices = LDSParams->LDSToReplacementIndicesMap[GV];
      uint32_t Idx0 = Indices[0];
      uint32_t Idx1 = Indices[1];
      uint32_t Idx2 = Indices[2];
      Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, Idx0),
                            ConstantInt::get(Int32Ty, Idx1),
                            ConstantInt::get(Int32Ty, Idx2)};
      Constant *GEP = ConstantExpr::getGetElementPtr(
          MallocMetadataStructType, MallocMetadataGlobal, GEPIdx, true);
      auto elt = ConstantExpr::getPtrToInt(GEP, Int32Ty);
      Elements.push_back(elt);
    } else
      Elements.push_back(PoisonValue::get(Int32Ty));
  }
  return ConstantArray::get(KernelOffsetsType, Elements);
}

void AMDGPUSwLowerLDS::BuildNonKernelLDSBaseTable(
    std::shared_ptr<NonKernelLDSParameters> &NKLDSParams) {
  // Base table will have single row, with elements of the row
  // placed as per kernel ID. Each element in the row corresponds
  // to addresss of malloc LDS global variable of the kernel.
  auto &Kernels = NKLDSParams->OrderedKernels;
  assert(!Kernels.empty());
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  const size_t NumberKernels = Kernels.size();
  ArrayType *AllKernelsOffsetsType = ArrayType::get(Int32Ty, NumberKernels);
  std::vector<Constant *> overallConstantExprElts(NumberKernels);
  for (size_t i = 0; i < NumberKernels; i++) {
    Function *Func = Kernels[i];
    auto &LDSParams = KernelToLDSParametersMap[Func];
    assert(LDSParams);
    GlobalVariable *MallocLDSGlobal = LDSParams->MallocLDSGlobal;
    assert(MallocLDSGlobal);
    Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, 0)};
    Constant *GEP = ConstantExpr::getGetElementPtr(
        MallocLDSGlobal->getType(), MallocLDSGlobal, GEPIdx, true);
    auto Elt = ConstantExpr::getPtrToInt(GEP, Int32Ty);
    overallConstantExprElts[i] = Elt;
  }
  Constant *init =
      ConstantArray::get(AllKernelsOffsetsType, overallConstantExprElts);
  NKLDSParams->LDSBaseTable = new GlobalVariable(
      M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, init,
      "llvm.amdgcn.sw.lds.base.table", nullptr, GlobalValue::NotThreadLocal,
      AMDGPUAS::CONSTANT_ADDRESS);
}

void AMDGPUSwLowerLDS::BuildNonKernelLDSOffsetTable(
    std::shared_ptr<NonKernelLDSParameters> &NKLDSParams) {
  // Offset table will have multiple rows and columns.
  // Rows are assumed to be from 0 to (n-1). n is total number
  // of kernels accessing the LDS through non-kernels.
  // Each row will have m elements. m is the total number of
  // unique LDS globals accessed by non-kernels.
  // Each element in the row correspond to the address of
  // the replacement of LDS global done by that particular kernel.
  auto &Variables = NKLDSParams->OrdereLDSGlobals;
  auto &Kernels = NKLDSParams->OrderedKernels;
  assert(!Variables.empty());
  assert(!Kernels.empty());
  LLVMContext &Ctx = M.getContext();
  const size_t NumberVariables = Variables.size();
  const size_t NumberKernels = Kernels.size();

  ArrayType *KernelOffsetsType =
      ArrayType::get(Type::getInt32Ty(Ctx), NumberVariables);

  ArrayType *AllKernelsOffsetsType =
      ArrayType::get(KernelOffsetsType, NumberKernels);
  // Constant *Missing = PoisonValue::get(KernelOffsetsType);
  std::vector<Constant *> overallConstantExprElts(NumberKernels);
  for (size_t i = 0; i < NumberKernels; i++) {
    Function *Func = Kernels[i];
    overallConstantExprElts[i] =
        GetAddressesOfVariablesInKernel(Func, Variables);
  }
  Constant *init =
      ConstantArray::get(AllKernelsOffsetsType, overallConstantExprElts);
  NKLDSParams->LDSOffsetTable = new GlobalVariable(
      M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, init,
      "llvm.amdgcn.sw.lds.offset.table", nullptr, GlobalValue::NotThreadLocal,
      AMDGPUAS::CONSTANT_ADDRESS);
  return;
}

void AMDGPUSwLowerLDS::LowerNonKernelLDSAccesses(
    Function *Func, SetVector<GlobalVariable *> &LDSGlobals,
    std::shared_ptr<NonKernelLDSParameters> &NKLDSParams) {
  // Replace LDS access in non-kernel with replacement queried from
  // Base table and offset from offset table.
  auto *EntryBlock = &Func->getEntryBlock();
  IRB.SetInsertPoint(EntryBlock, EntryBlock->begin());
  Function *Decl =
      Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_lds_kernel_id, {});
  auto *KernelId = IRB.CreateCall(Decl, {});
  GlobalVariable *LDSBaseTable = NKLDSParams->LDSBaseTable;
  GlobalVariable *LDSOffsetTable = NKLDSParams->LDSOffsetTable;
  auto &OrdereLDSGlobals = NKLDSParams->OrdereLDSGlobals;
  assert(LDSBaseTable && LDSOffsetTable);
  Value *BaseGEP = IRB.CreateInBoundsGEP(
      LDSBaseTable->getValueType(), LDSBaseTable, {IRB.getInt32(0), KernelId});
  Value *BaseLoad = IRB.CreateLoad(IRB.getInt32Ty(), BaseGEP);

  for (GlobalVariable *GV : LDSGlobals) {
    Value *BasePtr = IRB.CreateIntToPtr(BaseLoad, GV->getType());
    auto GVIt = std::find(OrdereLDSGlobals.begin(), OrdereLDSGlobals.end(), GV);
    assert(GVIt != OrdereLDSGlobals.end());
    uint32_t GVOffset = std::distance(OrdereLDSGlobals.begin(), GVIt);
    Value *OffsetGEP = IRB.CreateInBoundsGEP(
        LDSOffsetTable->getValueType(), LDSOffsetTable,
        {IRB.getInt32(0), KernelId, IRB.getInt32(GVOffset)});
    Value *OffsetLoad = IRB.CreateLoad(IRB.getInt32Ty(), OffsetGEP);
    OffsetLoad = IRB.CreateIntToPtr(OffsetLoad, GV->getType());
    OffsetLoad = IRB.CreateLoad(IRB.getInt32Ty(), OffsetLoad);
    Value *BasePlusOffset =
        IRB.CreateInBoundsGEP(GV->getType(), BasePtr, {OffsetLoad});
    ReplacesUsesOfGlobalInFunction(Func, GV, BasePlusOffset);
  }
  return;
}

static void ReorderStaticDynamicIndirectLDSSet(
    std::shared_ptr<KernelLDSParameters> &LDSParams) {
  // Sort Static, dynamic LDS globals which are either
  // direct or indirect access on basis of name.
  assert(LDSParams);
  LDSParams->DirectAccess.StaticLDSGlobals =
      SortByName(std::vector<GlobalVariable *>(
          LDSParams->DirectAccess.StaticLDSGlobals.begin(),
          LDSParams->DirectAccess.StaticLDSGlobals.end()));
  LDSParams->DirectAccess.DynamicLDSGlobals =
      SortByName(std::vector<GlobalVariable *>(
          LDSParams->DirectAccess.DynamicLDSGlobals.begin(),
          LDSParams->DirectAccess.DynamicLDSGlobals.end()));
  LDSParams->IndirectAccess.StaticLDSGlobals =
      SortByName(std::vector<GlobalVariable *>(
          LDSParams->IndirectAccess.StaticLDSGlobals.begin(),
          LDSParams->IndirectAccess.StaticLDSGlobals.end()));
  LDSParams->IndirectAccess.DynamicLDSGlobals =
      SortByName(std::vector<GlobalVariable *>(
          LDSParams->IndirectAccess.DynamicLDSGlobals.begin(),
          LDSParams->IndirectAccess.DynamicLDSGlobals.end()));
  return;
}

bool AMDGPUSwLowerLDS::Run() {
  bool Changed = false;
  CallGraph CG = CallGraph(M);
  SetVector<Function *> KernelsWithIndirectLDSAccess;
  FunctionVariableMap NonKernelToLDSAccessMap;
  SetVector<GlobalVariable *> AllNonKernelLDSAccess;

  Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

  // Get all the direct and indirect access of LDS for all the kernels.
  LDSUsesInfoTy LDSUsesInfo = getTransitiveUsesOfLDS(CG, M);

  // Get the Uses of LDS from non-kernels.
  GetUsesOfLDSByNonKernels(CG, NonKernelToLDSAccessMap);

  // Utility to group LDS access into direct, indirect, static and dynamic.
  auto PopulateKernelStaticDynamicLDS = [&](FunctionVariableMap &LDSAccesses,
                                            bool DirectAccess) {
    for (auto &K : LDSAccesses) {
      Function *F = K.first;
      assert(isKernelLDS(F));
      assert(!K.second.empty());

      if (!KernelToLDSParametersMap.contains(F))
        KernelToLDSParametersMap[F] = std::make_shared<KernelLDSParameters>();

      auto &LDSParams = KernelToLDSParametersMap[F];
      assert(LDSParams);
      if (!DirectAccess)
        KernelsWithIndirectLDSAccess.insert(F);
      for (GlobalVariable *GV : K.second) {
        if (!DirectAccess) {
          if (AMDGPU::isDynamicLDS(*GV))
            LDSParams->IndirectAccess.DynamicLDSGlobals.insert(GV);
          else
            LDSParams->IndirectAccess.StaticLDSGlobals.insert(GV);
          AllNonKernelLDSAccess.insert(GV);
        } else {
          if (AMDGPU::isDynamicLDS(*GV))
            LDSParams->DirectAccess.DynamicLDSGlobals.insert(GV);
          else
            LDSParams->DirectAccess.StaticLDSGlobals.insert(GV);
        }
      }
    }
  };

  PopulateKernelStaticDynamicLDS(LDSUsesInfo.direct_access, true);
  PopulateKernelStaticDynamicLDS(LDSUsesInfo.indirect_access, false);

  for (auto &K : KernelToLDSParametersMap) {
    Function *Func = K.first;
    auto &LDSParams = KernelToLDSParametersMap[Func];
    assert(LDSParams);
    if (LDSParams->DirectAccess.StaticLDSGlobals.empty() &&
        LDSParams->DirectAccess.DynamicLDSGlobals.empty() &&
        LDSParams->IndirectAccess.StaticLDSGlobals.empty() &&
        LDSParams->IndirectAccess.DynamicLDSGlobals.empty()) {
      Changed = false;
    } else {
      ReorderStaticDynamicIndirectLDSSet(LDSParams);
      PopulateMallocLDSGlobal(Func);
      PopulateMallocMetadataGlobal(Func);
      PopulateLDSToReplacementIndicesMap(Func);
      DomTreeUpdater DTU(DTCallback(*Func),
                         DomTreeUpdater::UpdateStrategy::Lazy);
      LowerKernelLDSAccesses(Func, DTU);
      Changed = true;
    }
  }

  std::shared_ptr<NonKernelLDSParameters> NKLDSParams =
      std::make_shared<NonKernelLDSParameters>();
  if (!NonKernelToLDSAccessMap.empty()) {
    assert(NKLDSParams);
    NKLDSParams->OrderedKernels = GetOrderedIndirectLDSAccessingKernels(
        std::move(KernelsWithIndirectLDSAccess));
    NKLDSParams->OrdereLDSGlobals =
        GetOrderedNonKernelAllLDSGlobals(std::move(AllNonKernelLDSAccess));
    assert(!NKLDSParams->OrderedKernels.empty());
    assert(!NKLDSParams->OrdereLDSGlobals.empty());
    BuildNonKernelLDSBaseTable(NKLDSParams);
    BuildNonKernelLDSOffsetTable(NKLDSParams);
    for (auto &K : NonKernelToLDSAccessMap) {
      Function *Func = K.first;
      DenseSet<GlobalVariable *> &LDSGlobals = K.second;
      SetVector<GlobalVariable *> OrderedLDSGlobals = SortByName(
          std::vector<GlobalVariable *>(LDSGlobals.begin(), LDSGlobals.end()));
      LowerNonKernelLDSAccesses(Func, OrderedLDSGlobals, NKLDSParams);
    }
  }
  return Changed;
}

class AMDGPUSwLowerLDSLegacy : public ModulePass {
public:
  static char ID;
  AMDGPUSwLowerLDSLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
} // namespace

char AMDGPUSwLowerLDSLegacy::ID = 0;
char &llvm::AMDGPUSwLowerLDSLegacyPassID = AMDGPUSwLowerLDSLegacy::ID;

INITIALIZE_PASS(AMDGPUSwLowerLDSLegacy, "amdgpu-sw-lower-lds",
                "AMDGPU Software lowering of LDS", false, false)

bool AMDGPUSwLowerLDSLegacy::runOnModule(Module &M) {
  DominatorTreeWrapperPass *const DTW =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto DTCallback = [&DTW](Function &F) -> DominatorTree * {
    return DTW ? &DTW->getDomTree() : nullptr;
  };
  bool IsChanged = false;
  AMDGPUSwLowerLDS SwLowerLDSImpl(M, DTCallback);
  IsChanged |= SwLowerLDSImpl.Run();
  return IsChanged;
}

ModulePass *llvm::createAMDGPUSwLowerLDSLegacyPass() {
  return new AMDGPUSwLowerLDSLegacy();
}

PreservedAnalyses AMDGPUSwLowerLDSPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto DTCallback = [&FAM](Function &F) -> DominatorTree * {
    return &FAM.getResult<DominatorTreeAnalysis>(F);
  };
  bool IsChanged = false;
  AMDGPUSwLowerLDS SwLowerLDSImpl(M, DTCallback);
  IsChanged |= SwLowerLDSImpl.Run();
  if (!IsChanged)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
