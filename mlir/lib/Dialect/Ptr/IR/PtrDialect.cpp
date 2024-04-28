//===- PtrDialect.cpp - Pointer dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Pointer dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer dialect
//===----------------------------------------------------------------------===//

void PtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Pointer API.
//===----------------------------------------------------------------------===//

// Error constants for vector data types.
constexpr const static unsigned kInvalidRankError = -1;
constexpr const static unsigned kScalableDimsError = -2;

// Returns a pair containing:
// The underlying type of a vector or the type itself if it's not a vector.
// The number of elements in the vector or an error code if the type is not
// supported.
static std::pair<Type, int64_t> getVecOrScalarInfo(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty)) {
    auto elemTy = vecTy.getElementType();
    // Vectors of rank greater than one or with scalable dimensions are not
    // supported.
    if (vecTy.getRank() != 1)
      return {elemTy, kInvalidRankError};
    else if (vecTy.getScalableDims()[0])
      return {elemTy, kScalableDimsError};
    return {elemTy, vecTy.getShape()[0]};
  }
  // `ty` is a scalar type.
  return {ty, 0};
}

/// Checks whether the shape of the operands is compatible with the operation.
/// Operands must be scalars or have the same vector shape, additionally only
/// vectors of rank 1 are supported.
static LogicalResult verifyShapeInfo(mlir::Operation *op,
                                     const std::pair<Type, int64_t> &tgtInfo,
                                     const std::pair<Type, int64_t> &srcInfo) {
  // Check shape validity.
  if (tgtInfo.second == kInvalidRankError ||
      srcInfo.second == kInvalidRankError)
    return op ? op->emitError("vectors of rank != 1 are not supported")
              : failure();
  if (tgtInfo.second == kScalableDimsError ||
      srcInfo.second == kScalableDimsError)
    return op ? op->emitError(
                    "vectors with scalable dimensions are not supported")
              : failure();
  if (tgtInfo.second != srcInfo.second)
    return op ? op->emitError("incompatible operand shapes") : failure();
  return success();
}

LogicalResult mlir::ptr::isValidAddrSpaceCastImpl(Type tgt, Type src,
                                                  Operation *op) {
  std::pair<Type, int64_t> tgtInfo = getVecOrScalarInfo(tgt);
  std::pair<Type, int64_t> srcInfo = getVecOrScalarInfo(src);
  if (!isa<PtrType>(tgtInfo.first) || !isa<PtrType>(srcInfo.first))
    return op ? op->emitError("invalid ptr-like operand") : failure();
  // Verify shape validity.
  return verifyShapeInfo(op, tgtInfo, srcInfo);
}

LogicalResult mlir::ptr::isValidPtrIntCastImpl(Type intLikeTy, Type ptrLikeTy,
                                               Operation *op) {
  // Check int-like type.
  std::pair<Type, int64_t> intInfo = getVecOrScalarInfo(intLikeTy);
  // The int-like operand is invalid.
  if (!intInfo.first.isSignlessIntOrIndex())
    return op ? op->emitError("invalid int-like type") : failure();
  // Check ptr-like type.
  std::pair<Type, int64_t> ptrInfo = getVecOrScalarInfo(ptrLikeTy);
  // The pointer-like operand is invalid.
  if (!isa<PtrType>(ptrInfo.first))
    return op ? op->emitError("invalid ptr-like type") : failure();
  // Verify shape validity.
  return verifyShapeInfo(op, intInfo, ptrInfo);
}

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
