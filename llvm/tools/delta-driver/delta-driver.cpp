//===-- delta-driver.cpp - Tool to drive Automated Delta Debugging --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See the llvm-project/llvm/docs/ProgrammersManual.rst to see how to use this
// tool
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/Program.h"

using namespace llvm;

cl::opt<std::string> ReproductionCmd(cl::Positional, cl::Required);

cl::opt<std::string> StartChunks(cl::Positional, cl::Required);

cl::opt<bool> Pessimist("pessimist", cl::init(false));

SmallVector<Chunk> simplifyChunksList(const SmallVector<Chunk> &Chunks) {
  SmallVector<Chunk> Res;
  Res.push_back(Chunks.front());
  for (unsigned Idx = 1; Idx < Chunks.size(); Idx++) {
    if (Chunks[Idx].Begin == Res.back().End + 1)
      Res.back().End = Chunks[Idx].End;
    else
      Res.push_back(Chunks[Idx]);
  }
  return Res;
}

bool stillReproducesIssue(const SmallVector<Chunk> &Chunks) {
  SmallVector<Chunk> SimpleChunks = simplifyChunksList(Chunks);

  std::string ChunkStr;
  {
    raw_string_ostream OS(ChunkStr);
    printChunks(OS, SimpleChunks);
  }

  errs() << "Checking with: " << ChunkStr << "\n";

  std::vector<StringRef> Argv;
  Argv.push_back(ReproductionCmd);
  Argv.push_back(ChunkStr);

  std::string ErrMsg;
  bool ExecutionFailed;
  int Result = sys::ExecuteAndWait(Argv[0], Argv, std::nullopt, {}, 0, 0,
                                   &ErrMsg, &ExecutionFailed);
  if (ExecutionFailed) {
    errs() << "failed to execute : " << Argv[0] << " : " << ErrMsg << "\n";
    exit(1);
  }

  bool Res = Result != 0;
  if (Res) {
    errs() << "SUCCESS : Still Interesting\n";
  } else {
    errs() << "FAILURE : Not Interesting\n";
  }
  return Res;
}

static bool increaseGranularity(SmallVector<Chunk> &Chunks) {
  errs() << "Increasing granularity\n";
  SmallVector<Chunk> NewChunks;
  bool SplitOne = false;

  for (auto &C : Chunks) {
    if (C.Begin == C.End)
      NewChunks.push_back(C);
    else {
      int Half = (C.Begin + C.End) / 2;
      NewChunks.push_back({C.Begin, Half});
      NewChunks.push_back({Half + 1, C.End});
      SplitOne = true;
    }
  }
  if (SplitOne) {
    Chunks = std::move(NewChunks);
  }
  return SplitOne;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  SmallVector<Chunk> CurrChunks;
  if (parseChunks(StartChunks, CurrChunks)) {
    return 1;
  }

  auto Program = sys::findProgramByName(ReproductionCmd);
  if (!Program) {
    errs() << "failed to find command : " << ReproductionCmd << "\n";
    return 1;
  }
  ReproductionCmd.setValue(Program.get());

  errs() << "Input Checking:\n";
  if (!stillReproducesIssue(CurrChunks)) {
    errs() << "starting chunks are not interesting\n";
    return 1;
  }
  if (CurrChunks.size() == 1)
    increaseGranularity(CurrChunks);
  if (Pessimist)
    while (increaseGranularity(CurrChunks)) {
    }
  while (1) {
    SmallDenseSet<unsigned> NotNeedChunks;
    auto FilteredCopy = [&]() {
      SmallVector<Chunk> CopiedChunks;
      for (unsigned SubIdx = 0; SubIdx < CurrChunks.size(); SubIdx++)
        if (!NotNeedChunks.count(SubIdx))
          CopiedChunks.push_back(CurrChunks[SubIdx]);
      return CopiedChunks;
    };

    for (int Idx = (CurrChunks.size() - 1); Idx >= 0; Idx--) {
      if (NotNeedChunks.size() + 1 == CurrChunks.size())
        break;

      errs() << "Trying to remove : ";
      CurrChunks[Idx].print(errs());
      errs() << "\n";

      NotNeedChunks.insert(Idx);
      SmallVector<Chunk> NextChunks = FilteredCopy();
      if (!stillReproducesIssue(NextChunks)) {
        NotNeedChunks.erase(Idx);
      }
    }
    CurrChunks = FilteredCopy();
    bool HasSplit = increaseGranularity(CurrChunks);
    if (!HasSplit)
      break;
  }

  errs() << "Minimal Chunks = ";
  printChunks(llvm::errs(), simplifyChunksList(CurrChunks));
  errs() << "\n";
}
