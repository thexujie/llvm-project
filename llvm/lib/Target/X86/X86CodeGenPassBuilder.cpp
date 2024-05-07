//===-- X86CodeGenPassBuilder.cpp ---------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains X86 CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"

#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"

using namespace llvm;

namespace {

class X86CodeGenPassBuilder
    : public CodeGenPassBuilder<X86CodeGenPassBuilder, X86TargetMachine> {
public:
  explicit X86CodeGenPassBuilder(X86TargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassBuilder &PB)
      : CodeGenPassBuilder(TM, Opts, PB) {}
  void addPreISel();
  void addAsmPrinter(CreateMCStreamer);
  Error addInstSelector();
};

void X86CodeGenPassBuilder::addPreISel() {
  // TODO: Add passes pre instruction selection.
}

void X86CodeGenPassBuilder::addAsmPrinter(CreateMCStreamer) {
  // TODO: Add AsmPrinter.
}

Error X86CodeGenPassBuilder::addInstSelector() {
  // TODO: Add instruction selector.
  return Error::success();
}

} // namespace

Error X86TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, const CGPassBuilderOption &Opt, PassBuilder &PB) {
  auto CGPB = X86CodeGenPassBuilder(*this, Opt, PB);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}
