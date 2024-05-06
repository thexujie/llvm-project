//==- RISCVSchedXiangShanKunMingHu.td - XiangShanKunMingHu Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The XiangShan is a high-performance open-source RISC-V processor project 
// initiated by the Institute of Computing Technology(ICT), Chinese Academy of Sciences(CAS). 
// The KunMingHu architecture is its third-generation derivative, 
// developed by the Institute of Computing Technology, Chinese Academy of Sciences  
// and the Beijing Institute of Open Source Chip (BOSC), 
// with a focus on achieving higher performance.
// Source: https://github.com/OpenXiangShan/XiangShan
// Documentation: https://github.com/OpenXiangShan/XiangShan-doc

//===----------------------------------------------------------------------===//
// KunMingHu core supports "RV64IMAFDCV_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh
// _zksed_zksh_svinval_zicbom_zicboz_zicsr_zifencei"
// then floating-point SEW can only be 64 and 32, not 16 and 8.
class NoZvfhSchedSEWSet_rm8and16<string mx, bit isF = 0, bit isWidening = 0> {
  defvar t = SchedSEWSet<mx, isF, isWidening>.val; 
  defvar remove8and16 = !if(isF, !listremove(t, [8, 16]), t);
  list<int> val = remove8and16;
}

class NoZvfhSmallestSEW<string mx, bit isF = 0, bit isWidening = 0> {
  int r = !head(NoZvfhSchedSEWSet_rm8and16<mx, isF, isWidening>.val);
}

multiclass NoZvfh_LMULSEWReadAdvanceImpl<string name, int val, list<SchedWrite> writes = [],
                                  list<string> MxList, bit isF = 0,
                                  bit isWidening = 0> {
  if !exists<SchedRead>(name # "_WorstCase") then
    def : ReadAdvance<!cast<SchedRead>(name # "_WorstCase"), val, writes>;
  foreach mx = MxList in {
    foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF, isWidening>.val in
      if !exists<SchedRead>(name # "_" # mx # "_E" # sew) then
        def : ReadAdvance<!cast<SchedRead>(name # "_" # mx # "_E" # sew), val, writes>;
  }
}

multiclass LMULSEWReadAdvanceFnoZvfh<string name, int val, list<SchedWrite> writes = []>
  : NoZvfh_LMULSEWReadAdvanceImpl<name, val, writes, SchedMxListF, isF=1,
                           isWidening=0>;

multiclass LMULSEWReadAdvanceFWnoZvfh<string name, int val, list<SchedWrite> writes = []>
    : NoZvfh_LMULSEWReadAdvanceImpl<name, val, writes, SchedMxListFW, isF = 1,
                             isWidening=1>;

//===----------------------------------------------------------------------===//
// If Zvfhmin and Zvfh are not supported, floating-point SEW can only be 32 or 64.
class NoZvfhSchedSEWSet_rm32and64<string mx, bit isF = 0, bit isWidening = 0> {
  defvar t = SchedSEWSet<mx, isF, isWidening>.val;
  defvar remove32and64 = !if(isF, !listremove(t, [32, 64]), t);
  list<int> val = remove32and64;
}

// Write-Impl
multiclass NoZvfhLMULSEWWriteResImpl<string name, list<ProcResourceKind> resources,
                               list<string> MxList, bit isF = 0,
                               bit isWidening = 0> {
  foreach mx = MxList in {
    foreach sew = NoZvfhSchedSEWSet_rm32and64<mx, isF, isWidening>.val in
      if !exists<SchedWrite>(name # "_" # mx # "_E" # sew) then
        def : WriteRes<!cast<SchedWrite>(name # "_" # mx # "_E" # sew), resources>;
  }
}
// Read-Impl
multiclass NoZvfhLMULSEWReadAdvanceImpl<string name, int val, list<SchedWrite> writes = [],
                                  list<string> MxList, bit isF = 0,
                                  bit isWidening = 0> {
  foreach mx = MxList in {
    foreach sew = NoZvfhSchedSEWSet_rm32and64<mx, isF, isWidening>.val in
      if !exists<SchedRead>(name # "_" # mx # "_E" # sew) then
        def : ReadAdvance<!cast<SchedRead>(name # "_" # mx # "_E" # sew), val, writes>;
  }
}

// Write
multiclass NoZvfhLMULSEWWriteResF<string name, list<ProcResourceKind> resources>
    : NoZvfhLMULSEWWriteResImpl<name, resources, SchedMxListF, isF=1>;

multiclass NoZvfhLMULSEWWriteResFW<string name, list<ProcResourceKind> resources>
    : NoZvfhLMULSEWWriteResImpl<name, resources, SchedMxListFW, isF=1, isWidening=1>;

multiclass NoZvfhLMULSEWWriteResFWRed<string name, list<ProcResourceKind> resources>
    : NoZvfhLMULSEWWriteResImpl<name, resources, SchedMxListFWRed, isF=1, isWidening=1>;

// Read
multiclass NoZvfhLMULSEWReadAdvanceF<string name, int val, list<SchedWrite> writes = []>
  : NoZvfhLMULSEWReadAdvanceImpl<name, val, writes, SchedMxListF, isF=1>;
multiclass
    NoZvfhLMULSEWReadAdvanceFW<string name, int val, list<SchedWrite> writes = []>
    : NoZvfhLMULSEWReadAdvanceImpl<name, val, writes, SchedMxListFW, isF=1,
                             isWidening = 1>;

multiclass UnsupportedSchedZvfh {
let Unsupported = true in {
// Write 
// 13. Vector Floating-Point Instructions
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFALUV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFALUF", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWALUV", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWALUF", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFMulV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFMulF", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFDivV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFDivF", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWMulV", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWMulF", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFMulAddV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFMulAddF", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWMulAddV", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWMulAddF", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFSqrtV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFRecpV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFMinMaxV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFMinMaxF", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFSgnjV", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFSgnjF", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFCvtIToFV", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFWCvtFToFV", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFNCvtIToFV", []>;
defm "" : NoZvfhLMULSEWWriteResFW<"WriteVFNCvtFToFV", []>;

// 14. Vector Reduction Operations
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFRedV_From", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFRedOV_From", []>;
defm "" : NoZvfhLMULSEWWriteResF<"WriteVFRedMinMaxV_From", []>;
defm "" : NoZvfhLMULSEWWriteResFWRed<"WriteVFWRedV_From", []>;
defm "" : NoZvfhLMULSEWWriteResFWRed<"WriteVFWRedOV_From", []>;

// Read
// 13. Vector Floating-Point Instructions
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFALUV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFALUF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFMulV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFMulF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFDivV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFDivF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFMulAddV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFMulAddF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFRecpV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFMinMaxV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFMinMaxF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFSgnjV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFSgnjF", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceF<"ReadVFCvtIToFV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : NoZvfhLMULSEWReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

} // Unsupported
} // UnsupportedSchedZvfh

//===----------------------------------------------------------------------===//

class XSGetCyclesVIALU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 2,
    !eq(mx, "MF8") : 2
  );
}

class XSGetCyclesVIMAC<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 3,
    !eq(mx, "M2") : 6,
    !eq(mx, "M4") : 12,
    !eq(mx, "M8") : 24,
    !eq(mx, "MF2") : 3,
    !eq(mx, "MF4") : 3,
    !eq(mx, "MF8") : 3
  );
}

class XSGetCyclesVIDIV<string mx, int sew> {
  int uop = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 2,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
  int cycles = !cond(
    !eq(sew, 64) : 19,   // I64: 4-19
    !eq(sew, 32) : 11,   // I32: 4-11
    !eq(sew, 16) : 7,    // I16: 4-7
    !eq(sew, 8) : 6      // I8: 6
  );
  int c = !mul(uop, cycles);
}

class XSGetCyclesVIPU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 2,
    !eq(mx, "MF8") : 2
  );    
}

class XSGetCyclesVPPU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 2,
    !eq(mx, "MF8") : 2
  );    
}

class XSGetCyclesVFALU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 2,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 2,
    !eq(mx, "MF8") : 2
  );    
}

class XSGetCyclesVFMA<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 8,
    !eq(mx, "M4") : 16,
    !eq(mx, "M8") : 32,
    !eq(mx, "MF2") : 4,
    !eq(mx, "MF4") : 4,
    !eq(mx, "MF8") : 4
  );    
}

class XSGetCyclesVFDIV<string mx, int sew> {
  assert !or(!eq(sew, 32), !eq(sew, 64)), "Floating-point SEW of KunMingHu can only be 32 or 64.";
  int uop = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 2,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
  int cycles = !cond(
    !eq(sew, 64) : 15,   // FP64: 15
    !eq(sew, 32) : 10,   // FP32: 10
  );
  int c = !mul(uop, cycles);
}

class XSGetCyclesVFCVT<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 3,
    !eq(mx, "M2") : 6,
    !eq(mx, "M4") : 12,
    !eq(mx, "M8") : 24,
    !eq(mx, "MF2") : 3,
    !eq(mx, "MF4") : 3,
    !eq(mx, "MF8") : 3
  );    
}

class XSGetCyclesVLDU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 8,
    !eq(mx, "M2") : 16,
    !eq(mx, "M4") : 32,
    !eq(mx, "M8") : 64,
    !eq(mx, "MF2") : 8,
    !eq(mx, "MF4") : 8,
    !eq(mx, "MF8") : 8
  );
}

class XSGetCyclesVSTU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 7,
    !eq(mx, "M2") : 14,
    !eq(mx, "M4") : 28,
    !eq(mx, "M8") : 56,
    !eq(mx, "MF2") : 7,
    !eq(mx, "MF4") : 7,
    !eq(mx, "MF8") : 7
  );
}

// If mx is the maximum LMUL in the MxList, then c is true, indicating the worst case.
class XSIsWorstCaseMX<string mx, list<string> MxList> {
  defvar LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

// If mx is the maximum LMUL in the MxList, and sew is the minimum value 
// when LMUL=mx, then c is true, indicating the worst case.
class XSIsWorstCaseMXSEW<string mx, int sew, list<string> MxList,
                               bit isF = 0> {
  defvar LLMUL = LargestLMUL<MxList>.r;
  defvar SSEW = NoZvfhSmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

class XSLDUtoAnyBypass<SchedRead read, int cycles = 2>
    : ReadAdvance<read, cycles, [WriteLDB, WriteLDH,
                                 WriteLDW, WriteLDD,
                                 WriteAtomicW, WriteAtomicD,
                                 WriteAtomicLDW, WriteAtomicLDD]>;

//===----------------------------------------------------------------------===//

def XiangShanKunMingHuModel : SchedMachineModel {
  let IssueWidth = 6;   // 6-way decode and dispatch
  let MicroOpBufferSize = 256;
  let LoopMicroOpBufferSize = 48;  // Instruction queue size
  let LoadLatency = 6;
  let MispredictPenalty = 13; // Based on estimate of pipeline depth.
  let PostRAScheduler = 1;
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtZcmt, HasStdExtZkr];
}

let SchedModel = XiangShanKunMingHuModel in {
// Define each kind of processor resource and number available.
/// Pipline
let BufferSize = 12 in {
  // Integer
  def XSPipeALU0 : ProcResource<1>; // ALU, MUL, BKU
  def XSPipeALU1 : ProcResource<1>; // ALU, MUL, BKU
  def XSPipeALU2 : ProcResource<1>; // ALU
  def XSPipeALU3 : ProcResource<1>; // ALU
  def XSPipeBJU0 : ProcResource<1>; // BRU, JMP
  def XSPipeBJU1 : ProcResource<1>; // BRU, JMP
  def XSPipeBJU2 : ProcResource<1>; // BRU, JMP, I2F, I2V, VSET, CSR, FENCE
  def XSPipeDIV  : ProcResource<1>; // DIV

  // Vector and floating-point
  def XSPipVFEX0 : ProcResource<1>; // VFALU, VFMA, VIALU, VIMAC
  def XSPipVFEX1 : ProcResource<1>; // VIPU, VPPU, VFCVT, F2V, VSET2
  def XSPipVFEX2 : ProcResource<1>; // VFALU, VFMA, VIALU
  def XSPipVFEX3 : ProcResource<1>; // VFDIV, VIDIV

  // Vector load and store
  def XSPipVLDU  : ProcResource<1>; // VLDU
  def XSPipVSTU  : ProcResource<1>; // VSTU
}

let BufferSize = 24 in {
  // Load and store
  def XSPipeLDU0 : ProcResource<1>; // LDU
  def XSPipeLDU1 : ProcResource<1>; // LDU
  def XSPipeLDU2 : ProcResource<1>; // LDU
  def XSPipeSTU0 : ProcResource<1>; // STU
  def XSPipeSTU1 : ProcResource<1>; // STU
}

def XSPipeGroupALU : ProcResGroup<[XSPipeALU0, XSPipeALU1, XSPipeALU2, XSPipeALU3]>;
def XSPipeGroupMUL : ProcResGroup<[XSPipeALU0, XSPipeALU1]>;
def XSPipeGroupBKU : ProcResGroup<[XSPipeALU0, XSPipeALU1]>;
def XSPipeGroupBRU : ProcResGroup<[XSPipeBJU0, XSPipeBJU1, XSPipeBJU2]>;
def XSPipeGroupJMP : ProcResGroup<[XSPipeBJU0, XSPipeBJU1, XSPipeBJU2]>;

def XSPipeGroupVIALU : ProcResGroup<[XSPipVFEX0, XSPipVFEX2]>;
def XSPipeGroupVFALU : ProcResGroup<[XSPipVFEX0, XSPipVFEX2]>;
def XSPipeGroupVFMA  : ProcResGroup<[XSPipVFEX0, XSPipVFEX2]>;

def XSPipeGroupLDU : ProcResGroup<[XSPipeLDU0, XSPipeLDU1, XSPipeLDU2]>;
def XSPipeGroupSTU : ProcResGroup<[XSPipeSTU0, XSPipeSTU1]>;

/// Register
def XS_INT_PRF : RegisterFile<224, [GPR], [1], [1], 0, 0>;
def XS_FP_PRF  : RegisterFile<192, [FPR64], [1], [1], 0, 0>;

//===----------------------------------------------------------------------===//

// Jump
let Latency = 1 in {
  def : WriteRes<WriteJmp, [XSPipeGroupBRU]>;
  def : WriteRes<WriteJal, [XSPipeGroupJMP]>;
  def : WriteRes<WriteJalr, [XSPipeGroupJMP]>;
}

// Integer arithmetic and logic
let Latency = 1 in {
  def : WriteRes<WriteIALU32, [XSPipeGroupALU]>;
  def : WriteRes<WriteIALU, [XSPipeGroupALU]>;
  def : WriteRes<WriteShiftImm32, [XSPipeGroupALU]>;
  def : WriteRes<WriteShiftImm, [XSPipeGroupALU]>;
  def : WriteRes<WriteShiftReg32, [XSPipeGroupALU]>;
  def : WriteRes<WriteShiftReg, [XSPipeGroupALU]>;
}

// Integer multiplication
let Latency = 3 in {
  def : WriteRes<WriteIMul, [XSPipeGroupMUL]>;
  def : WriteRes<WriteIMul32, [XSPipeGroupMUL]>;
}

// Integer division
// Worst case latency is used.
// The latency of integer division ranges from 4 to 20.
let Latency = 20, ReleaseAtCycles = [20] in {
  def : WriteRes<WriteIDiv32, [XSPipeDIV]>;
  def : WriteRes<WriteIDiv, [XSPipeDIV]>;
  def : WriteRes<WriteIRem32, [XSPipeDIV]>;
  def : WriteRes<WriteIRem, [XSPipeDIV]>;
}

// Memory
let Latency = 5 in {
  def : WriteRes<WriteSTB, [XSPipeGroupSTU]>;
  def : WriteRes<WriteSTH, [XSPipeGroupSTU]>;
  def : WriteRes<WriteSTW, [XSPipeGroupSTU]>;
  def : WriteRes<WriteSTD, [XSPipeGroupSTU]>;
  def : WriteRes<WriteFST32, [XSPipeGroupSTU]>;
  def : WriteRes<WriteFST64, [XSPipeGroupSTU]>;
  def : WriteRes<WriteAtomicSTW, [XSPipeGroupSTU]>;
  def : WriteRes<WriteAtomicSTD, [XSPipeGroupSTU]>;
}
let Latency = 6 in {
  def : WriteRes<WriteLDB, [XSPipeGroupLDU]>;
  def : WriteRes<WriteLDH, [XSPipeGroupLDU]>;
  def : WriteRes<WriteLDW, [XSPipeGroupLDU]>;
  def : WriteRes<WriteLDD, [XSPipeGroupLDU]>;
  def : WriteRes<WriteFLD32, [XSPipeGroupLDU]>;
  def : WriteRes<WriteFLD64, [XSPipeGroupLDU]>;
  def : WriteRes<WriteAtomicW, [XSPipeGroupLDU]>;
  def : WriteRes<WriteAtomicD, [XSPipeGroupLDU]>;
  def : WriteRes<WriteAtomicLDW, [XSPipeGroupLDU]>;
  def : WriteRes<WriteAtomicLDD, [XSPipeGroupLDU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFAdd32, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFAdd64, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFCmp32, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFCmp64, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFMinMax32, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFMinMax64, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFClass32, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFClass64, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFSGNJ32, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteFSGNJ64, [XSPipeGroupVFALU]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFMul32, [XSPipeGroupVFMA]>;
  def : WriteRes<WriteFMul64, [XSPipeGroupVFMA]>;
  def : WriteRes<WriteFMA32, [XSPipeGroupVFMA]>;
  def : WriteRes<WriteFMA64, [XSPipeGroupVFMA]>;
}

// VFDIV
let Latency = 10 in {
  def : WriteRes<WriteFDiv32, [XSPipVFEX3]>;
  def : WriteRes<WriteFSqrt32, [XSPipVFEX3]>;
}
let Latency = 15 in {
  def : WriteRes<WriteFDiv64, [XSPipVFEX3]>;
  def : WriteRes<WriteFSqrt64, [XSPipVFEX3]>;
}

// VFCVT
let Latency = 3 in {
  def : WriteRes<WriteFCvtF32ToI32, [XSPipVFEX1]>;
  def : WriteRes<WriteFCvtF32ToI64, [XSPipVFEX1]>;
  def : WriteRes<WriteFCvtF64ToI32, [XSPipVFEX1]>;
  def : WriteRes<WriteFCvtF64ToI64, [XSPipVFEX1]>;
  def : WriteRes<WriteFCvtF64ToF32, [XSPipVFEX1]>;
  def : WriteRes<WriteFCvtF32ToF64, [XSPipVFEX1]>;
  def : WriteRes<WriteFMovF64ToI64, [XSPipVFEX1]>;
  def : WriteRes<WriteFMovF32ToI32, [XSPipVFEX1]>;
}

// I2V
let Latency = 1 in {
  def : WriteRes<WriteFMovI64ToF64, [XSPipeBJU2]>;
  def : WriteRes<WriteFMovI32ToF32, [XSPipeBJU2]>;
}

// I2F
let Latency = 3 in {
  def : WriteRes<WriteFCvtI32ToF32, [XSPipeBJU2]>;
  def : WriteRes<WriteFCvtI64ToF32, [XSPipeBJU2]>;
  def : WriteRes<WriteFCvtI32ToF64, [XSPipeBJU2]>;
  def : WriteRes<WriteFCvtI64ToF64, [XSPipeBJU2]>;
}

/// Zb*
let Latency = 1 in {
  // Zba
  def : WriteRes<WriteSHXADD, [XSPipeGroupALU]>;
  def : WriteRes<WriteSHXADD32, [XSPipeGroupALU]>;
  
  // Zbb
  def : WriteRes<WriteRotateImm, [XSPipeGroupALU]>;
  def : WriteRes<WriteRotateImm32, [XSPipeGroupALU]>;
  def : WriteRes<WriteRotateReg, [XSPipeGroupALU]>;
  def : WriteRes<WriteRotateReg32, [XSPipeGroupALU]>;
  def : WriteRes<WriteREV8, [XSPipeGroupALU]>;
  def : WriteRes<WriteORCB, [XSPipeGroupALU]>;
  def : WriteRes<WriteIMinMax, [XSPipeGroupALU]>;

  // Zbs
  def : WriteRes<WriteSingleBit, [XSPipeGroupALU]>;
  def : WriteRes<WriteSingleBitImm, [XSPipeGroupALU]>;
  def : WriteRes<WriteBEXT, [XSPipeGroupALU]>;
  def : WriteRes<WriteBEXTI, [XSPipeGroupALU]>;

  // Zbkb
  def : WriteRes<WriteBREV8, [XSPipeGroupALU]>;
  def : WriteRes<WritePACK, [XSPipeGroupALU]>;
  def : WriteRes<WritePACK32, [XSPipeGroupALU]>;
  def : WriteRes<WriteZIP, [XSPipeGroupALU]>;
}

let Latency = 3 in {
  // Zbb
  def : WriteRes<WriteCLZ, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCLZ32, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCTZ, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCTZ32, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCPOP, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCPOP32, [XSPipeGroupBKU]>;

  // Zbc
  def : WriteRes<WriteCLMUL, [XSPipeGroupBKU]>;

  // Zbkx
  def : WriteRes<WriteXPERM, [XSPipeGroupBKU]>;
}

/// Vector extension
// 3.6 Vector Byte Length vlenb
def : WriteRes<WriteRdVLENB, [XSPipeGroupALU]>;

// 6. Configuration-Setting Instructions
// VSET VSET2
let Latency = 1 in {
  def : WriteRes<WriteVSETVLI, [XSPipVFEX1]>;
  def : WriteRes<WriteVSETIVLI, [XSPipVFEX1]>;
  def : WriteRes<WriteVSETVL, [XSPipeBJU2]>;
}

// 7. Vector Loads and Stores
// VLDU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVLDU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVLDE",       [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDM",       [XSPipVLDU], mx, IsWorstCase>;
  }
}

// VSTU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVSTU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVSTE",       [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTM",       [XSPipVSTU], mx, IsWorstCase>;
  }
}

// VLDU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVLDU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVLDS8",      [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS16",     [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS32",     [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS64",     [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX8",     [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX16",    [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX32",    [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX64",    [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8",     [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16",    [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32",    [XSPipVLDU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64",    [XSPipVLDU], mx, IsWorstCase>;
  }
}

// VSTU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVSTU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVSTS8",      [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS16",     [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS32",     [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS64",     [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX8",     [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16",    [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32",    [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64",    [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8",     [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16",    [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32",    [XSPipVSTU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64",    [XSPipVSTU], mx, IsWorstCase>;
  }
}

// VLDU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVLDU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVLDFF",      [XSPipVLDU], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar CyclesLoad = XSGetCyclesVLDU<mx>.c;
      defvar CyclesStore = XSGetCyclesVSTU<mx>.c;
      defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
      let Latency = CyclesLoad in {
        // VLDU
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [XSPipVLDU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [XSPipVLDU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [XSPipVLDU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [XSPipVLDU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [XSPipVLDU], mx, IsWorstCase>;
      }
      let Latency = CyclesStore in {
        // VSTU
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [XSPipVSTU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [XSPipVSTU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [XSPipVSTU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [XSPipVSTU], mx, IsWorstCase>;
      }
    }
  }
}

// VLDU
let Latency = 8 in 
  def : WriteRes<WriteVLD1R, [XSPipVLDU]>;
let Latency = 16 in 
  def : WriteRes<WriteVLD2R, [XSPipVLDU]>;
let Latency = 32 in 
  def : WriteRes<WriteVLD4R, [XSPipVLDU]>;
let Latency = 64 in 
  def : WriteRes<WriteVLD8R, [XSPipVLDU]>;

// VSTU
let Latency = 7 in 
  def : WriteRes<WriteVST1R, [XSPipVSTU]>;
let Latency = 14 in 
  def : WriteRes<WriteVST2R, [XSPipVSTU]>;
let Latency = 28 in 
  def : WriteRes<WriteVST4R, [XSPipVSTU]>;
let Latency = 56 in 
  def : WriteRes<WriteVST8R, [XSPipVSTU]>;

// 11. Vector Integer Arithmetic Instructions
// VIALU
// The latency of KunMingHu vector extension instructions is independent of SEW.
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVIALUV",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVExtV",      [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUV",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftV",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxV",  [XSPipeGroupVIALU], mx, IsWorstCase>;
  }

  // Because .vx and .vi need to be converted to .vv before execution, 
  // an additional cycle is required.
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVIALUX",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX",  [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
}

// VIALU
foreach mx = SchedMxListW in {
  defvar Cycles = XSGetCyclesVIALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftV",   [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX",   [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI",   [XSPipeGroupVIALU], mx, IsWorstCase>;
   
  }
}

// VIMAC
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIMAC<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVIMulV",     [XSPipVFEX0], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVIMulX",     [XSPipVFEX0], mx, IsWorstCase>;
  }
}

// VIDIV
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = XSGetCyclesVIDIV<mx, sew>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV",   [XSPipVFEX3], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX",   [XSPipVFEX3], mx, sew, IsWorstCase>;
    }
  }
}

// VIMAC
foreach mx = SchedMxListW in {
  defvar Cycles = XSGetCyclesVIMAC<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [XSPipVFEX0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [XSPipVFEX0], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [XSPipVFEX0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [XSPipVFEX0], mx, IsWorstCase>;
  }
}

// VIMAC
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIMAC<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVIMulAddV",  [XSPipVFEX0], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVIMulAddX",  [XSPipVFEX0], mx, IsWorstCase>;
  }
}

// VIALU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVIMergeV",   [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUV",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",     [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVIMergeX",   [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI",   [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI",     [XSPipeGroupVIALU], mx, IsWorstCase>;

    // 12. Vector Fixed-Point Arithmetic Instructions
    defm "" : LMULWriteResMX<"WriteVSALUX",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",     [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",     [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
}

// VIMAC
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIMAC<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVSMulV",     [XSPipVFEX0], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVSMulX",     [XSPipVFEX0], mx, IsWorstCase>;
  }
}

// VIALU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVSShiftV",   [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVSShiftX",   [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI",   [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
}

// VIALU
foreach mx = SchedMxListW in {
  defvar Cycles = XSGetCyclesVIALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVNClipV",    [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVNClipX",    [XSPipeGroupVIALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",    [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
// VFALU
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFALU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",     [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",     [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
  }
}

// VFALU
foreach mx = SchedMxListFW in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVFALU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV",    [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF",    [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
  }
}

// VFMA
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFMA<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",     [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",     [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
  }
}

// VFDIV
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFDIV<mx, sew>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",   [XSPipVFEX3], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",   [XSPipVFEX3], mx, sew, IsWorstCase>;
    }
  }
}

// VFMA
foreach mx = SchedMxListFW in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVFMA<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV",    [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF",    [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
  }
}

// VFMA
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFMA<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV",  [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF",  [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
  }
}

// VFMA
foreach mx = SchedMxListFW in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVFMA<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [XSPipeGroupVFMA], mx, sew, IsWorstCase>;
    }
  }
}

// VFDIV
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFDIV<mx, sew>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV",    [XSPipVFEX3], mx, sew, IsWorstCase>;
    }
  }
}

// VFCVT
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFCVT<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV",    [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VFALU
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFALU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV",  [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV",    [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
    let Latency = !add(Cycles, 1) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF",  [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF",    [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
  }
}

// VFALU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVFALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",     [XSPipeGroupVFALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFClassV",   [XSPipeGroupVFALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",   [XSPipeGroupVFALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",     [XSPipeGroupVFALU], mx, IsWorstCase>;
  }
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVFCmpF",     [XSPipeGroupVFALU], mx, IsWorstCase>;
  }
}

// VFCVT
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFCVT<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV",    [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VFCVT
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVFCVT<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVFCvtFToIV",    [XSPipVFEX1], mx, IsWorstCase>;
  }
}

// VFCVT
foreach mx = SchedMxListW in {
    foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVFCVT<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV",   [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VFCVT
foreach mx = SchedMxListFW in {
  defvar Cycles = XSGetCyclesVFCVT<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVFWCvtFToIV",   [XSPipVFEX1], mx, IsWorstCase>;
  }
}

// VFCVT
foreach mx = SchedMxListFW in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVFCVT<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV",   [XSPipVFEX1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV",   [XSPipVFEX1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV",   [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VFCVT
foreach mx = SchedMxListW in {
  defvar Cycles = XSGetCyclesVFCVT<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV",   [XSPipVFEX1], mx, IsWorstCase>;
  }
}

// 14. Vector Reduction Operations
// VIPU
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = XSGetCyclesVIPU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From",        [XSPipVFEX1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From",  [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VIPU
foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVIPU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From",       [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VFALU
foreach mx = SchedMxListF in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=0>.val in {
    defvar Cycles = XSGetCyclesVFALU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From",        [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From",       [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From",  [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
  }
}

// VFALU
foreach mx = SchedMxListFWRed in {
  foreach sew = NoZvfhSchedSEWSet_rm8and16<mx, isF=1, isWidening=1>.val in {
    defvar Cycles = XSGetCyclesVFALU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, isF=1>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",       [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From",      [XSPipeGroupVFALU], mx, sew, IsWorstCase>;
    }
  }
}

// 15. Vector Mask Instructions
// VIALU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIALU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVMALUV",   [XSPipeGroupVIALU], mx, IsWorstCase>;
  }
}

// VIPU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVIPU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = Cycles in {
    defm "" : LMULWriteResMX<"WriteVMPopV",   [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV",   [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV",   [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIotaV",   [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIdxV",    [XSPipVFEX1], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
let Latency = 2 in {
  // VIALU
  def : WriteRes<WriteVMovSX, [XSPipVFEX0]>;
  // VIPU
  def : WriteRes<WriteVMovXS, [XSPipVFEX1]>;
}

// VFALU
let Latency = 2 in {
  def : WriteRes<WriteVMovSF, [XSPipeGroupVFALU]>;
  def : WriteRes<WriteVMovFS, [XSPipeGroupVFALU]>;
}

// VPPU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVPPU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVISlideX",   [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlideI",   [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlide1X",  [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F",  [XSPipVFEX1], mx, IsWorstCase>;
  }
}

// VPPU
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = XSGetCyclesVPPU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV",  [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VPPU
foreach mx = SchedMxList in {
  defvar Cycles = XSGetCyclesVPPU<mx>.c;
  defvar IsWorstCase = XSIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(Cycles, 1) in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [XSPipVFEX1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [XSPipVFEX1], mx, IsWorstCase>;
  }
}

// VPPU
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar Cycles = XSGetCyclesVPPU<mx>.c;
    defvar IsWorstCase = XSIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = Cycles in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV",  [XSPipVFEX1], mx, sew, IsWorstCase>;
    }
  }
}

// VPPU
let Latency = 2 in 
  def : WriteRes<WriteVMov1V,     [XSPipVFEX1]>;
let Latency = 4 in   
  def : WriteRes<WriteVMov2V,     [XSPipVFEX1]>;
let Latency = 6 in 
  def : WriteRes<WriteVMov4V,     [XSPipVFEX1]>;
let Latency = 8 in 
  def : WriteRes<WriteVMov8V,     [XSPipVFEX1]>;

// Others
def : WriteRes<WriteCSR, [XSPipeBJU2]>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//

// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : XSLDUtoAnyBypass<ReadIALU>;
def : XSLDUtoAnyBypass<ReadIALU32>;
def : XSLDUtoAnyBypass<ReadShiftImm>;
def : XSLDUtoAnyBypass<ReadShiftImm32>;
def : XSLDUtoAnyBypass<ReadShiftReg>;
def : XSLDUtoAnyBypass<ReadShiftReg32>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

/// B extension
// Zba
def : XSLDUtoAnyBypass<ReadSHXADD>;
def : XSLDUtoAnyBypass<ReadSHXADD32>;
// Zbb
def : XSLDUtoAnyBypass<ReadRotateImm>;
def : XSLDUtoAnyBypass<ReadRotateImm32>;
def : XSLDUtoAnyBypass<ReadRotateReg>;
def : XSLDUtoAnyBypass<ReadRotateReg32>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : XSLDUtoAnyBypass<ReadREV8>;
def : XSLDUtoAnyBypass<ReadORCB>;
def : XSLDUtoAnyBypass<ReadIMinMax>;
// Zbc
def : ReadAdvance<ReadCLMUL, 0>;
// Zbs
def : XSLDUtoAnyBypass<ReadSingleBit>;
def : XSLDUtoAnyBypass<ReadSingleBitImm>;
// Zbkb
def : XSLDUtoAnyBypass<ReadBREV8>;
def : XSLDUtoAnyBypass<ReadPACK>;
def : XSLDUtoAnyBypass<ReadPACK32>;
def : XSLDUtoAnyBypass<ReadZIP>;
// Zbkx
def : ReadAdvance<ReadXPERM, 0>;

/// V extension
// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 0>;
def : ReadAdvance<ReadVSETVL, 0>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
// These are already LMUL aware
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 11. Vector Integer Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVIALUV", 0>;
defm "" : LMULReadAdvance<"ReadVIALUX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm "" : LMULReadAdvance<"ReadVExtV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUX", 0>;
defm "" : LMULReadAdvance<"ReadVShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm "" : LMULReadAdvance<"ReadVICmpV", 0>;
defm "" : LMULReadAdvance<"ReadVICmpX", 0>;
defm "" : LMULReadAdvance<"ReadVIMinMaxV", 0>;
defm "" : LMULReadAdvance<"ReadVIMinMaxX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulX", 0>;
defm "" : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVIDivX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovX", 0>;

// 12. Vector Fixed-Point Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;

// 13. Vector Floating-Point Instructions
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFALUV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFALUF", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWALUV", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWALUF", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFMulV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFMulF", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFDivF", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWMulV", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWMulF", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFMulAddV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFMulAddF", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWMulAddV", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFSqrtV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFRecpV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFMinMaxV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFMinMaxF", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFSgnjV", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULSEWReadAdvanceFnoZvfh<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULSEWReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFWCvtFToFV", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULSEWReadAdvanceFWnoZvfh<"ReadVFNCvtFToFV", 0>;

// 14. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFRedMinMaxV, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 15. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVIotaV", 0>;

// 16. Vector Permutation Instructions
def : ReadAdvance<ReadVMovXS, 0>;
def : ReadAdvance<ReadVMovSX_V, 0>;
def : ReadAdvance<ReadVMovSX_X, 0>;
def : ReadAdvance<ReadVMovFS, 0>;
def : ReadAdvance<ReadVMovSF_V, 0>;
def : ReadAdvance<ReadVMovSF_F, 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULReadAdvance<"ReadVGatherV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
// These are already LMUL aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;
def : ReadAdvance<ReadVMergeOp_WorstCase, 0>;
foreach mx = SchedMxList in {
  def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx), 0>;
  foreach sew = SchedSEWSet<mx>.val in
    def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx  # "_E" # sew), 0>;
}

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfh;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZvfh;

// Move Elimination
def : IsOptimizableRegisterMove<[
  InstructionEquivalenceClass<[C_MV], TruePred >
]>;

} // SchedModel
