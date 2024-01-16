//==---- DynamicAllocator.cpp - Types for the constexpr VM -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicAllocator.h"
#include "InterpBlock.h"
#include "InterpState.h"

using namespace clang;
using namespace clang::interp;

Block *DynamicAllocator::allocate(const Expr *Source, PrimType T,
                                  size_t NumElements) {
  assert(NumElements > 0);
  // Create a new descriptor for an array of the specified size and
  // element type.
  const Descriptor *D = allocateDescriptor(
      Source, T, Descriptor::InlineDescMD, NumElements, /*IsConst=*/false,
      /*IsTemporary=*/false, /*IsMutable=*/false);
  return allocate(D);
}

Block *DynamicAllocator::allocate(const Descriptor *ElementDesc,
                                  size_t NumElements) {
  assert(NumElements > 0);
  // Create a new descriptor for an array of the specified size and
  // element type.
  const Descriptor *D = allocateDescriptor(
      ElementDesc->asExpr(), ElementDesc, Descriptor::InlineDescMD, NumElements,
      /*IsConst=*/false, /*IsTemporary=*/false, /*IsMutable=*/false);
  return allocate(D);
}

Block *DynamicAllocator::allocate(const Descriptor *D) {
  assert(D->asExpr());

  auto Memory =
      std::make_unique<std::byte[]>(sizeof(Block) + D->getAllocSize());
  auto *B = new (Memory.get()) Block(D, /*isStatic=*/false);
  B->invokeCtor();

  InlineDescriptor *ID = reinterpret_cast<InlineDescriptor *>(B->rawData());
  ID->Desc = const_cast<Descriptor *>(D);
  ID->IsActive = true;
  ID->Offset = sizeof(InlineDescriptor);
  ID->IsBase = false;
  ID->IsFieldMutable = false;
  ID->IsConst = false;
  ID->IsInitialized = false;
  assert(ID->Desc);

  if (auto It = AllocationSites.find(D->asExpr()); It != AllocationSites.end())
    It->second.Allocations.emplace_back(std::move(Memory));
  else
    AllocationSites.insert(
        {D->asExpr(), AllocationSite(std::move(Memory), D->isArray())});
  return B;
}

bool DynamicAllocator::deallocate(const Expr *Source,
                                  const Block *BlockToDelete, InterpState &S) {
  auto It = AllocationSites.find(Source);
  if (It == AllocationSites.end())
    return false;

  auto &Site = It->second;
  assert(Site.size() > 0);

  // Find the Block to delete.
  auto AllocIt = llvm::find_if(Site.Allocations, [&](const Allocation &A) {
    const Block *B = reinterpret_cast<const Block *>(A.Memory.get());
    return BlockToDelete == B;
  });

  assert(AllocIt != Site.Allocations.end());

  Block *B = reinterpret_cast<Block *>(AllocIt->Memory.get());
  B->invokeDtor();
  S.deallocate(B);
  Site.Allocations.erase(AllocIt);

  if (Site.size() == 0)
    AllocationSites.erase(It);

  return true;
}
