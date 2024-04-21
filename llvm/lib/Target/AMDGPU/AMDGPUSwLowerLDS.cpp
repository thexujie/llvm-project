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
//    "SW LDS" in this pass.
//    Each LDS access corresponds to an offset in the allocated memory.
//    All static LDS accesses will be allocated first and then dynamic LDS
//    will occupy the device global memory.
//    To store the offsets corresponding to all LDS accesses, another global
//    variable is created which will be called "SW LDS metadata" in this pass.
//    - SW LDS Global:
//        It is LDS global of ptr type with name
//        "llvm.amdgcn.sw.lds.<kernel-name>".
//    - Metadata Global:
//        It is of struct type, with n members. n equals the number of LDS
//        globals accessed by the kernel(direct and indirect). Each member of
//        struct is another struct of type {i32, i32, i32}. First member
//        corresponds to offset, second member corresponds to size of LDS global
//        being replaced and third represents the total aligned size. It will
//        have name "llvm.amdgcn.sw.lds.<kernel-name>.md". This global will have
//        an intializer with static LDS related offsets and sizes initialized.
//        But for dynamic LDS related entries, offsets will be intialized to
//        previous static LDS allocation end offset. Sizes for them will be zero
//        initially. These dynamic LDS offset and size values will be updated
//        with in the kernel, since kernel can read the dynamic LDS size
//        allocation done at runtime with query to "hidden_dynamic_lds_size"
//        hidden kernel argument.
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
//        to addresss of "SW LDS" variable created for
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
//    from base and offset tables. Based on kernel-id query, address of "SW
//    LDS" for that corresponding kernel is obtained from base table.
//    The Offset into the base "SW LDS" is obtained from
//    corresponding element in offset table. With this information, replacement
//    value is obtained.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
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
#include "llvm/Support/raw_ostream.h"
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
  GlobalVariable *SwLDS = nullptr;
  GlobalVariable *SwLDSMetadata = nullptr;
  LDSAccessTypeInfo DirectAccess;
  LDSAccessTypeInfo IndirectAccess;
  DenseMap<GlobalVariable *, SmallVector<uint32_t, 3>>
      LDSToReplacementIndicesMap;
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
  AMDGPUSwLowerLDS(Module &Mod, DomTreeCallback Callback)
      : M(Mod), IRB(M.getContext()), DTCallback(Callback) {}
  bool run();
  void getUsesOfLDSByNonKernels(const CallGraph &CG,
                                FunctionVariableMap &Functions);
  SetVector<Function *>
  getOrderedIndirectLDSAccessingKernels(SetVector<Function *> &&Kernels);
  SetVector<GlobalVariable *>
  getOrderedNonKernelAllLDSGlobals(SetVector<GlobalVariable *> &&Variables);
  void populateSwLDSGlobal(Function *Func);
  void populateSwMetadataGlobal(Function *Func);
  void populateLDSToReplacementIndicesMap(Function *Func);
  void replaceKernelLDSAccesses(Function *Func);
  void lowerKernelLDSAccesses(Function *Func, DomTreeUpdater &DTU);
  void buildNonKernelLDSOffsetTable(NonKernelLDSParameters &NKLDSParams);
  void buildNonKernelLDSBaseTable(NonKernelLDSParameters &NKLDSParams);
  Constant *
  getAddressesOfVariablesInKernel(Function *Func,
                                  SetVector<GlobalVariable *> &Variables);
  void lowerNonKernelLDSAccesses(Function *Func,
                                 SetVector<GlobalVariable *> &LDSGlobals,
                                 NonKernelLDSParameters &NKLDSParams);

private:
  Module &M;
  IRBuilder<> IRB;
  DomTreeCallback DTCallback;
  DenseMap<Function *, KernelLDSParameters> KernelToLDSParametersMap;
};

template <typename T> SetVector<T> sortByName(std::vector<T> &&V) {
  // Sort the vector of globals or Functions based on their name.
  // Returns a SetVector of globals/Functions.
  sort(V, [](const auto *L, const auto *R) {
    return L->getName() < R->getName();
  });
  return {SetVector<T>(V.begin(), V.end())};
}

SetVector<GlobalVariable *> AMDGPUSwLowerLDS::getOrderedNonKernelAllLDSGlobals(
    SetVector<GlobalVariable *> &&Variables) {
  // Sort all the non-kernel LDS accesses based on their name.
  return sortByName(
      std::vector<GlobalVariable *>(Variables.begin(), Variables.end()));
}

SetVector<Function *> AMDGPUSwLowerLDS::getOrderedIndirectLDSAccessingKernels(
    SetVector<Function *> &&Kernels) {
  // Sort the non-kernels accessing LDS based on their name.
  // Also assign a kernel ID metadata based on the sorted order.
  LLVMContext &Ctx = M.getContext();
  if (Kernels.size() > UINT32_MAX) {
    // 32 bit keeps it in one SGPR. > 2**32 kernels won't fit on the GPU
    report_fatal_error("Unimplemented SW LDS lowering for > 2**32 kernels");
  }
  SetVector<Function *> OrderedKernels =
      sortByName(std::vector<Function *>(Kernels.begin(), Kernels.end()));
  for (size_t i = 0; i < Kernels.size(); i++) {
    Metadata *AttrMDArgs[1] = {
        ConstantAsMetadata::get(IRB.getInt32(i)),
    };
    Function *Func = OrderedKernels[i];
    Func->setMetadata("llvm.amdgcn.lds.kernel.id",
                      MDNode::get(Ctx, AttrMDArgs));
    auto &LDSParams = KernelToLDSParametersMap[Func];
  }
  return std::move(OrderedKernels);
}

void AMDGPUSwLowerLDS::getUsesOfLDSByNonKernels(
    const CallGraph &CG, FunctionVariableMap &functions) {
  // Get uses from the current function, excluding uses by called functions
  // Two output variables to avoid walking the globals list twice
  for (auto &GV : M.globals()) {
    if (!AMDGPU::isLDSVariableToLower(GV))
      continue;

    if (GV.isAbsoluteSymbolRef()) {
      report_fatal_error(
          "LDS variables with absolute addresses are unimplemented.");
    }

    for (User *V : GV.users()) {
      if (auto *I = dyn_cast<Instruction>(V)) {
        Function *F = I->getFunction();
        if (!isKernelLDS(F))
          functions[F].insert(&GV);
      }
    }
  }
}

void AMDGPUSwLowerLDS::populateSwLDSGlobal(Function *Func) {
  // Create new LDS global required for each kernel to store
  // device global memory pointer.
  auto &LDSParams = KernelToLDSParametersMap[Func];
  // create new global pointer variable
  LDSParams.SwLDS = new GlobalVariable(
      M, IRB.getPtrTy(), false, GlobalValue::InternalLinkage,
      PoisonValue::get(IRB.getPtrTy()), "llvm.amdgcn.sw.lds." + Func->getName(),
      nullptr, GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
  return;
}

void AMDGPUSwLowerLDS::populateSwMetadataGlobal(Function *Func) {
  // Create new metadata global for every kernel and initialize the
  // start offsets and sizes corresponding to each LDS accesses.
  auto &LDSParams = KernelToLDSParametersMap[Func];
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  std::vector<Type *> Items;
  Type *Int32Ty = IRB.getInt32Ty();
  std::vector<Constant *> Initializers;
  Align MaxAlignment(1);
  auto UpdateMaxAlignment = [&MaxAlignment, &DL](GlobalVariable *GV) {
    Align GVAlign = AMDGPU::getAlign(DL, GV);
    MaxAlignment = std::max(MaxAlignment, GVAlign);
  };

  for (GlobalVariable *GV : LDSParams.DirectAccess.StaticLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams.DirectAccess.DynamicLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams.IndirectAccess.StaticLDSGlobals)
    UpdateMaxAlignment(GV);

  for (GlobalVariable *GV : LDSParams.IndirectAccess.DynamicLDSGlobals)
    UpdateMaxAlignment(GV);

  //{StartOffset, AlignedSizeInBytes}
  SmallString<128> MDItemStr;
  raw_svector_ostream MDItemOS(MDItemStr);
  MDItemOS << "llvm.amdgcn.sw.lds." << Func->getName().str() << ".md.item";

  StructType *LDSItemTy =
      StructType::create(Ctx, {Int32Ty, Int32Ty, Int32Ty}, MDItemOS.str());
  uint32_t MallocSize = 0;
  auto buildInitializerForSwLDSMD =
      [&](SetVector<GlobalVariable *> &LDSGlobals) {
        for (auto &GV : LDSGlobals) {
          Type *Ty = GV->getValueType();
          const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
          Items.push_back(LDSItemTy);
          Constant *ItemStartOffset = ConstantInt::get(Int32Ty, MallocSize);
          Constant *SizeInBytesConst = ConstantInt::get(Int32Ty, SizeInBytes);
          uint64_t AlignedSize = alignTo(SizeInBytes, MaxAlignment);
          Constant *AlignedSizeInBytesConst =
              ConstantInt::get(Int32Ty, AlignedSize);
          MallocSize += AlignedSize;
          Constant *InitItem =
              ConstantStruct::get(LDSItemTy, {ItemStartOffset, SizeInBytesConst,
                                              AlignedSizeInBytesConst});
          Initializers.push_back(InitItem);
        }
      };

  buildInitializerForSwLDSMD(LDSParams.DirectAccess.StaticLDSGlobals);
  buildInitializerForSwLDSMD(LDSParams.IndirectAccess.StaticLDSGlobals);
  buildInitializerForSwLDSMD(LDSParams.DirectAccess.DynamicLDSGlobals);
  buildInitializerForSwLDSMD(LDSParams.IndirectAccess.DynamicLDSGlobals);

  SmallString<128> MDTypeStr;
  raw_svector_ostream MDTypeOS(MDTypeStr);
  MDTypeOS << "llvm.amdgcn.sw.lds." << Func->getName().str() << ".md.type";

  StructType *MetadataStructType =
      StructType::create(Ctx, Items, MDTypeOS.str());
  SmallString<128> MDStr;
  raw_svector_ostream MDOS(MDStr);
  MDOS << "llvm.amdgcn.sw.lds." << Func->getName().str() << ".md";
  LDSParams.SwLDSMetadata = new GlobalVariable(
      M, MetadataStructType, false, GlobalValue::InternalLinkage,
      PoisonValue::get(MetadataStructType), MDOS.str(), nullptr,
      GlobalValue::NotThreadLocal, AMDGPUAS::GLOBAL_ADDRESS, false);
  Constant *data = ConstantStruct::get(MetadataStructType, Initializers);
  LDSParams.SwLDSMetadata->setInitializer(data);
  assert(LDSParams.SwLDS);
  // Set the alignment to MaxAlignment for SwLDS.
  LDSParams.SwLDS->setAlignment(MaxAlignment);
  GlobalValue::SanitizerMetadata MD;
  MD.NoAddress = true;
  LDSParams.SwLDSMetadata->setSanitizerMetadata(MD);
  return;
}

void AMDGPUSwLowerLDS::populateLDSToReplacementIndicesMap(Function *Func) {
  // Fill the corresponding LDS replacement indices for each LDS access
  // related to this kernel.
  auto &LDSParams = KernelToLDSParametersMap[Func];
  auto PopulateIndices = [&](SetVector<GlobalVariable *> &LDSGlobals,
                             uint32_t &Idx) {
    for (auto &GV : LDSGlobals) {
      LDSParams.LDSToReplacementIndicesMap[GV] = {0, Idx, 0};
      ++Idx;
    }
  };
  uint32_t Idx = 0;
  PopulateIndices(LDSParams.DirectAccess.StaticLDSGlobals, Idx);
  PopulateIndices(LDSParams.IndirectAccess.StaticLDSGlobals, Idx);
  PopulateIndices(LDSParams.DirectAccess.DynamicLDSGlobals, Idx);
  PopulateIndices(LDSParams.IndirectAccess.DynamicLDSGlobals, Idx);
  return;
}

static void replacesUsesOfGlobalInFunction(Function *Func, GlobalVariable *GV,
                                           Value *Replacement) {
  // Replace all uses of LDS global in this Function with a Replacement.
  auto ReplaceUsesLambda = [Func](const Use &U) -> bool {
    auto *V = U.getUser();
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      auto *Func1 = Inst->getParent()->getParent();
      if (Func == Func1)
        return true;
    }
    return false;
  };
  GV->replaceUsesWithIf(Replacement, ReplaceUsesLambda);
}

void AMDGPUSwLowerLDS::replaceKernelLDSAccesses(Function *Func) {
  auto &LDSParams = KernelToLDSParametersMap[Func];
  GlobalVariable *SwLDS = LDSParams.SwLDS;
  assert(SwLDS);
  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDSMetadata);
  StructType *SwLDSMetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());
  Type *Int32Ty = IRB.getInt32Ty();

  auto &IndirectAccess = LDSParams.IndirectAccess;
  auto &DirectAccess = LDSParams.DirectAccess;
  // Replace all uses of LDS global in this Function with a Replacement.
  auto ReplaceLDSGlobalUses = [&](SetVector<GlobalVariable *> &LDSGlobals) {
    for (auto &GV : LDSGlobals) {
      // Do not generate instructions if LDS access is in non-kernel
      // i.e indirect-access.
      if ((IndirectAccess.StaticLDSGlobals.contains(GV) ||
           IndirectAccess.DynamicLDSGlobals.contains(GV)) &&
          (!DirectAccess.StaticLDSGlobals.contains(GV) &&
           !DirectAccess.DynamicLDSGlobals.contains(GV)))
        continue;
      auto &Indices = LDSParams.LDSToReplacementIndicesMap[GV];
      assert(Indices.size() == 3);
      uint32_t Idx0 = Indices[0];
      uint32_t Idx1 = Indices[1];
      uint32_t Idx2 = Indices[2];
      Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, Idx0),
                            ConstantInt::get(Int32Ty, Idx1),
                            ConstantInt::get(Int32Ty, Idx2)};
      Constant *GEP = ConstantExpr::getGetElementPtr(
          SwLDSMetadataStructType, SwLDSMetadata, GEPIdx, true);
      Value *Load = IRB.CreateLoad(Int32Ty, GEP);
      Value *BasePlusOffset =
          IRB.CreateInBoundsGEP(IRB.getInt8Ty(), SwLDS, {Load});
      LLVM_DEBUG(dbgs() << "Sw LDS Lowering, Replacing LDS "
                        << GV->getName().str());
      replacesUsesOfGlobalInFunction(Func, GV, BasePlusOffset);
    }
  };
  ReplaceLDSGlobalUses(DirectAccess.StaticLDSGlobals);
  ReplaceLDSGlobalUses(IndirectAccess.StaticLDSGlobals);
  ReplaceLDSGlobalUses(DirectAccess.DynamicLDSGlobals);
  ReplaceLDSGlobalUses(IndirectAccess.DynamicLDSGlobals);
}

void AMDGPUSwLowerLDS::lowerKernelLDSAccesses(Function *Func,
                                              DomTreeUpdater &DTU) {
  LLVM_DEBUG(dbgs() << "Sw Lowering Kernel LDS for : "
                    << Func->getName().str());
  auto &LDSParams = KernelToLDSParametersMap[Func];
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

  GlobalVariable *SwLDS = LDSParams.SwLDS;
  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDS && SwLDSMetadata);
  StructType *MetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());

  // All work items will branch to PrevEntryBlock except {0,0,0} index
  // work item which will branch to malloc block.
  IRB.CreateCondBr(WIdzCond, MallocBlock, PrevEntryBlock);

  // Malloc block
  IRB.SetInsertPoint(MallocBlock, MallocBlock->begin());

  // If Dynamic LDS globals are accessed by the kernel,
  // Get the size of dyn lds from hidden dyn_lds_size kernel arg.
  // Update the corresponding metadata global entries for this dyn lds global.
  uint32_t MallocSize = 0;
  Value *CurrMallocSize;

  unsigned NumStaticLDS = LDSParams.DirectAccess.StaticLDSGlobals.size() +
                          LDSParams.IndirectAccess.StaticLDSGlobals.size();
  unsigned NumDynLDS = LDSParams.DirectAccess.DynamicLDSGlobals.size() +
                       LDSParams.IndirectAccess.DynamicLDSGlobals.size();

  if (NumStaticLDS) {
    auto *GEPForEndStaticLDSOffset = IRB.CreateInBoundsGEP(
        MetadataStructType, SwLDSMetadata,
        {IRB.getInt32(0), IRB.getInt32(NumStaticLDS - 1), IRB.getInt32(0)});

    auto *GEPForEndStaticLDSSize = IRB.CreateInBoundsGEP(
        MetadataStructType, SwLDSMetadata,
        {IRB.getInt32(0), IRB.getInt32(NumStaticLDS - 1), IRB.getInt32(2)});

    Value *EndStaticLDSOffset =
        IRB.CreateLoad(IRB.getInt64Ty(), GEPForEndStaticLDSOffset);
    Value *EndStaticLDSSize =
        IRB.CreateLoad(IRB.getInt64Ty(), GEPForEndStaticLDSSize);
    CurrMallocSize = IRB.CreateAdd(EndStaticLDSOffset, EndStaticLDSSize);
  } else
    CurrMallocSize = IRB.getInt64(MallocSize);

  if (NumDynLDS) {
    unsigned MaxAlignment = SwLDS->getAlignment();
    Value *MaxAlignValue = IRB.getInt64(MaxAlignment);
    Value *MaxAlignValueMinusOne = IRB.getInt64(MaxAlignment - 1);

    Value *ImplicitArg =
        IRB.CreateIntrinsic(Intrinsic::amdgcn_implicitarg_ptr, {}, {});
    Value *HiddenDynLDSSize = IRB.CreateInBoundsGEP(
        ImplicitArg->getType(), ImplicitArg, {IRB.getInt32(15)});

    auto MallocSizeCalcLambda =
        [&](SetVector<GlobalVariable *> &DynamicLDSGlobals) {
          for (GlobalVariable *DynGV : DynamicLDSGlobals) {
            auto &Indices = LDSParams.LDSToReplacementIndicesMap[DynGV];

            // Update the Offset metadata.
            auto *GEPForOffset = IRB.CreateInBoundsGEP(
                MetadataStructType, SwLDSMetadata,
                {IRB.getInt32(0), IRB.getInt32(Indices[1]), IRB.getInt32(0)});
            IRB.CreateStore(CurrMallocSize, GEPForOffset);

            // Get size from hidden dyn_lds_size argument of kernel
            // Update the size and Aligned Size metadata.
            auto *GEPForSize = IRB.CreateInBoundsGEP(
                MetadataStructType, SwLDSMetadata,
                {IRB.getInt32(0), IRB.getInt32(Indices[1]), IRB.getInt32(1)});
            Value *CurrDynLDSSize =
                IRB.CreateLoad(IRB.getInt64Ty(), HiddenDynLDSSize);
            IRB.CreateStore(CurrDynLDSSize, GEPForSize);

            auto *GEPForAlignedSize = IRB.CreateInBoundsGEP(
                MetadataStructType, SwLDSMetadata,
                {IRB.getInt32(0), IRB.getInt32(Indices[1]), IRB.getInt32(2)});
            Value *AlignedDynLDSSize =
                IRB.CreateAdd(CurrDynLDSSize, MaxAlignValueMinusOne);
            AlignedDynLDSSize =
                IRB.CreateUDiv(AlignedDynLDSSize, MaxAlignValue);
            AlignedDynLDSSize = IRB.CreateMul(AlignedDynLDSSize, MaxAlignValue);
            IRB.CreateStore(AlignedDynLDSSize, GEPForAlignedSize);

            // Update the Current Malloc Size
            CurrMallocSize = IRB.CreateAdd(CurrMallocSize, AlignedDynLDSSize);
          }
        };
    MallocSizeCalcLambda(LDSParams.DirectAccess.DynamicLDSGlobals);
    MallocSizeCalcLambda(LDSParams.IndirectAccess.DynamicLDSGlobals);
  }

  // Create a call to malloc function which does device global memory allocation
  // with size equals to all LDS global accesses size  in this kernel.
  FunctionCallee AMDGPUMallocFunc = M.getOrInsertFunction(
      StringRef("malloc"),
      FunctionType::get(IRB.getPtrTy(1), {IRB.getInt64Ty()}, false));
  Value *MCI = IRB.CreateCall(AMDGPUMallocFunc, {CurrMallocSize});

  // create store of malloc to new global
  IRB.CreateStore(MCI, SwLDS);

  // Create branch to PrevEntryBlock
  IRB.CreateBr(PrevEntryBlock);

  // Create wave-group barrier at the starting of Previous entry block
  Type *Int1Ty = IRB.getInt1Ty();
  IRB.SetInsertPoint(PrevEntryBlock, PrevEntryBlock->begin());
  auto *XYZCondPhi = IRB.CreatePHI(Int1Ty, 2, "xyzCond");
  XYZCondPhi->addIncoming(IRB.getInt1(0), WIdBlock);
  XYZCondPhi->addIncoming(IRB.getInt1(1), MallocBlock);

  IRB.CreateIntrinsic(Intrinsic::amdgcn_s_barrier, {}, {});

  replaceKernelLDSAccesses(Func);

  auto *CondFreeBlock = BasicBlock::Create(Ctx, "CondFree", Func);
  auto *FreeBlock = BasicBlock::Create(Ctx, "Free", Func);
  auto *EndBlock = BasicBlock::Create(Ctx, "End", Func);
  for (BasicBlock &BB : *Func) {
    if (!BB.empty()) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&BB.back())) {
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
  FunctionCallee AMDGPUFreeReturn = M.getOrInsertFunction(
      StringRef("free"),
      FunctionType::get(IRB.getVoidTy(), {IRB.getPtrTy()}, false));

  Value *MallocPtr = IRB.CreateLoad(IRB.getPtrTy(), SwLDS);
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
}

Constant *AMDGPUSwLowerLDS::getAddressesOfVariablesInKernel(
    Function *Func, SetVector<GlobalVariable *> &Variables) {
  Type *Int32Ty = IRB.getInt32Ty();
  auto &LDSParams = KernelToLDSParametersMap[Func];

  GlobalVariable *SwLDSMetadata = LDSParams.SwLDSMetadata;
  assert(SwLDSMetadata);
  auto *SwLDSMetadataStructType =
      cast<StructType>(SwLDSMetadata->getValueType());
  ArrayType *KernelOffsetsType = ArrayType::get(Int32Ty, Variables.size());

  SmallVector<Constant *> Elements;
  for (size_t i = 0; i < Variables.size(); i++) {
    GlobalVariable *GV = Variables[i];
    if (!LDSParams.LDSToReplacementIndicesMap.contains(GV)) {
      Elements.push_back(PoisonValue::get(Int32Ty));
      continue;
    }
    auto &Indices = LDSParams.LDSToReplacementIndicesMap[GV];
    uint32_t Idx0 = Indices[0];
    uint32_t Idx1 = Indices[1];
    uint32_t Idx2 = Indices[2];
    Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, Idx0),
                          ConstantInt::get(Int32Ty, Idx1),
                          ConstantInt::get(Int32Ty, Idx2)};
    Constant *GEP = ConstantExpr::getGetElementPtr(SwLDSMetadataStructType,
                                                   SwLDSMetadata, GEPIdx, true);
    auto elt = ConstantExpr::getPtrToInt(GEP, Int32Ty);
    Elements.push_back(elt);
  }
  return ConstantArray::get(KernelOffsetsType, Elements);
}

void AMDGPUSwLowerLDS::buildNonKernelLDSBaseTable(
    NonKernelLDSParameters &NKLDSParams) {
  // Base table will have single row, with elements of the row
  // placed as per kernel ID. Each element in the row corresponds
  // to addresss of "SW LDS" global of the kernel.
  auto &Kernels = NKLDSParams.OrderedKernels;
  Type *Int32Ty = IRB.getInt32Ty();
  const size_t NumberKernels = Kernels.size();
  ArrayType *AllKernelsOffsetsType = ArrayType::get(Int32Ty, NumberKernels);
  std::vector<Constant *> OverallConstantExprElts(NumberKernels);
  for (size_t i = 0; i < NumberKernels; i++) {
    Function *Func = Kernels[i];
    auto &LDSParams = KernelToLDSParametersMap[Func];
    GlobalVariable *SwLDS = LDSParams.SwLDS;
    assert(SwLDS);
    Constant *GEPIdx[] = {ConstantInt::get(Int32Ty, 0)};
    Constant *GEP =
        ConstantExpr::getGetElementPtr(SwLDS->getType(), SwLDS, GEPIdx, true);
    auto Elt = ConstantExpr::getPtrToInt(GEP, Int32Ty);
    OverallConstantExprElts[i] = Elt;
  }
  Constant *init =
      ConstantArray::get(AllKernelsOffsetsType, OverallConstantExprElts);
  NKLDSParams.LDSBaseTable = new GlobalVariable(
      M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, init,
      "llvm.amdgcn.sw.lds.base.table", nullptr, GlobalValue::NotThreadLocal,
      AMDGPUAS::CONSTANT_ADDRESS);
}

void AMDGPUSwLowerLDS::buildNonKernelLDSOffsetTable(
    NonKernelLDSParameters &NKLDSParams) {
  // Offset table will have multiple rows and columns.
  // Rows are assumed to be from 0 to (n-1). n is total number
  // of kernels accessing the LDS through non-kernels.
  // Each row will have m elements. m is the total number of
  // unique LDS globals accessed by non-kernels.
  // Each element in the row correspond to the address of
  // the replacement of LDS global done by that particular kernel.
  auto &Variables = NKLDSParams.OrdereLDSGlobals;
  auto &Kernels = NKLDSParams.OrderedKernels;
  assert(!Variables.empty());
  assert(!Kernels.empty());
  const size_t NumberVariables = Variables.size();
  const size_t NumberKernels = Kernels.size();

  ArrayType *KernelOffsetsType =
      ArrayType::get(IRB.getInt32Ty(), NumberVariables);

  ArrayType *AllKernelsOffsetsType =
      ArrayType::get(KernelOffsetsType, NumberKernels);
  std::vector<Constant *> overallConstantExprElts(NumberKernels);
  for (size_t i = 0; i < NumberKernels; i++) {
    Function *Func = Kernels[i];
    overallConstantExprElts[i] =
        getAddressesOfVariablesInKernel(Func, Variables);
  }
  Constant *Init =
      ConstantArray::get(AllKernelsOffsetsType, overallConstantExprElts);
  NKLDSParams.LDSOffsetTable = new GlobalVariable(
      M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, Init,
      "llvm.amdgcn.sw.lds.offset.table", nullptr, GlobalValue::NotThreadLocal,
      AMDGPUAS::CONSTANT_ADDRESS);
}

void AMDGPUSwLowerLDS::lowerNonKernelLDSAccesses(
    Function *Func, SetVector<GlobalVariable *> &LDSGlobals,
    NonKernelLDSParameters &NKLDSParams) {
  // Replace LDS access in non-kernel with replacement queried from
  // Base table and offset from offset table.
  LLVM_DEBUG(dbgs() << "Sw LDS lowering, lower non-kernel access for : "
                    << Func->getName().str());
  auto *EntryBlock = &Func->getEntryBlock();
  IRB.SetInsertPoint(EntryBlock, EntryBlock->begin());
  Function *Decl =
      Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_lds_kernel_id, {});
  auto *KernelId = IRB.CreateCall(Decl, {});
  GlobalVariable *LDSBaseTable = NKLDSParams.LDSBaseTable;
  GlobalVariable *LDSOffsetTable = NKLDSParams.LDSOffsetTable;
  auto &OrdereLDSGlobals = NKLDSParams.OrdereLDSGlobals;
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
        IRB.CreateInBoundsGEP(IRB.getInt8Ty(), BasePtr, {OffsetLoad});
    LLVM_DEBUG(dbgs() << "Sw LDS Lowering, Replace non-kernel LDS for "
                      << GV->getName().str());
    replacesUsesOfGlobalInFunction(Func, GV, BasePlusOffset);
  }
}

static void reorderStaticDynamicIndirectLDSSet(KernelLDSParameters &LDSParams) {
  // Sort Static, dynamic LDS globals which are either
  // direct or indirect access on basis of name.
  auto &DirectAccess = LDSParams.DirectAccess;
  auto &IndirectAccess = LDSParams.IndirectAccess;
  LDSParams.DirectAccess.StaticLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(DirectAccess.StaticLDSGlobals.begin(),
                                    DirectAccess.StaticLDSGlobals.end()));
  LDSParams.DirectAccess.DynamicLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(DirectAccess.DynamicLDSGlobals.begin(),
                                    DirectAccess.DynamicLDSGlobals.end()));
  LDSParams.IndirectAccess.StaticLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(IndirectAccess.StaticLDSGlobals.begin(),
                                    IndirectAccess.StaticLDSGlobals.end()));
  LDSParams.IndirectAccess.DynamicLDSGlobals = sortByName(
      std::vector<GlobalVariable *>(IndirectAccess.DynamicLDSGlobals.begin(),
                                    IndirectAccess.DynamicLDSGlobals.end()));
}

bool AMDGPUSwLowerLDS::run() {
  bool Changed = false;
  CallGraph CG = CallGraph(M);
  SetVector<Function *> KernelsWithIndirectLDSAccess;
  FunctionVariableMap NonKernelToLDSAccessMap;
  SetVector<GlobalVariable *> AllNonKernelLDSAccess;

  Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

  // Get all the direct and indirect access of LDS for all the kernels.
  LDSUsesInfoTy LDSUsesInfo = getTransitiveUsesOfLDS(CG, M);

  // Get the Uses of LDS from non-kernels.
  getUsesOfLDSByNonKernels(CG, NonKernelToLDSAccessMap);

  // Utility to group LDS access into direct, indirect, static and dynamic.
  auto PopulateKernelStaticDynamicLDS = [&](FunctionVariableMap &LDSAccesses,
                                            bool DirectAccess) {
    for (auto &K : LDSAccesses) {
      Function *F = K.first;
      assert(isKernelLDS(F));

      if (!KernelToLDSParametersMap.contains(F)) {
        KernelLDSParameters KernelLDSParams;
        KernelToLDSParametersMap[F] = KernelLDSParams;
      }

      auto &LDSParams = KernelToLDSParametersMap[F];
      if (!DirectAccess)
        KernelsWithIndirectLDSAccess.insert(F);
      for (GlobalVariable *GV : K.second) {
        if (!DirectAccess) {
          if (AMDGPU::isDynamicLDS(*GV))
            LDSParams.IndirectAccess.DynamicLDSGlobals.insert(GV);
          else
            LDSParams.IndirectAccess.StaticLDSGlobals.insert(GV);
          AllNonKernelLDSAccess.insert(GV);
        } else {
          if (AMDGPU::isDynamicLDS(*GV))
            LDSParams.DirectAccess.DynamicLDSGlobals.insert(GV);
          else
            LDSParams.DirectAccess.StaticLDSGlobals.insert(GV);
        }
      }
    }
  };

  PopulateKernelStaticDynamicLDS(LDSUsesInfo.direct_access, true);
  PopulateKernelStaticDynamicLDS(LDSUsesInfo.indirect_access, false);

  for (auto &K : KernelToLDSParametersMap) {
    Function *Func = K.first;
    auto &LDSParams = KernelToLDSParametersMap[Func];
    if (LDSParams.DirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.DirectAccess.DynamicLDSGlobals.empty() &&
        LDSParams.IndirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.IndirectAccess.DynamicLDSGlobals.empty()) {
      Changed = false;
    } else {
      removeFnAttrFromReachable(CG, Func, "amdgpu-no-workitem-id-x");
      removeFnAttrFromReachable(CG, Func, "amdgpu-no-workitem-id-y");
      removeFnAttrFromReachable(CG, Func, "amdgpu-no-workitem-id-z");
      reorderStaticDynamicIndirectLDSSet(LDSParams);
      populateSwLDSGlobal(Func);
      populateSwMetadataGlobal(Func);
      populateLDSToReplacementIndicesMap(Func);
      DomTreeUpdater DTU(DTCallback(*Func),
                         DomTreeUpdater::UpdateStrategy::Lazy);
      lowerKernelLDSAccesses(Func, DTU);
      Changed = true;
    }
  }

  NonKernelLDSParameters NKLDSParams;
  if (!NonKernelToLDSAccessMap.empty()) {
    NKLDSParams.OrderedKernels = getOrderedIndirectLDSAccessingKernels(
        std::move(KernelsWithIndirectLDSAccess));
    NKLDSParams.OrdereLDSGlobals =
        getOrderedNonKernelAllLDSGlobals(std::move(AllNonKernelLDSAccess));
    assert(!NKLDSParams.OrderedKernels.empty());
    assert(!NKLDSParams.OrdereLDSGlobals.empty());
    buildNonKernelLDSBaseTable(NKLDSParams);
    buildNonKernelLDSOffsetTable(NKLDSParams);
    for (auto &K : NonKernelToLDSAccessMap) {
      Function *Func = K.first;
      DenseSet<GlobalVariable *> &LDSGlobals = K.second;
      SetVector<GlobalVariable *> OrderedLDSGlobals = sortByName(
          std::vector<GlobalVariable *>(LDSGlobals.begin(), LDSGlobals.end()));
      lowerNonKernelLDSAccesses(Func, OrderedLDSGlobals, NKLDSParams);
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
  IsChanged |= SwLowerLDSImpl.run();
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
  IsChanged |= SwLowerLDSImpl.run();
  if (!IsChanged)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
