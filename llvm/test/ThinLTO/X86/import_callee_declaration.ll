; "-debug-only" requires asserts.
; REQUIRES: asserts
; RUN: rm -rf %t && split-file %s %t && cd %t

; Generate per-module summaries.
; RUN: opt -module-summary main.ll -o main.bc
; RUN: opt -module-summary lib.ll -o lib.bc

; Generate the combined summary and distributed indices.

; - For function import, set 'import-instr-limit' to 7 and fall back to import
;   function declarations.
; - In main.ll, function 'main' calls 'small_func' and 'large_func'. Both callees
;   are defined in lib.ll. 'small_func' has two indirect callees, one is smaller
;   and the other one is larger. Both callees of 'small_func' are defined in lib.ll.
; - Given the import limit, in main's combined summary, the import type of 'small_func'
;   and 'small_indirect_callee' will be 'definition', and the import type of
;   'large_func' and 'large_indirect_callee' will be 'declaration'.
;
; The test will disassemble combined summaries and check the import type is
; correct. Right now postlink optimizer pipeline doesn't do anything (e.g.,
; import the declaration or de-serialize summary attributes yet) so there is
; nothing to test more than the summary content.
;
; TODO: Extend this test case to test IR once postlink optimizer makes use of
; the import type for declarations, and add test coverage for in-process thinlto.
;
; RUN: llvm-lto2 run \
; RUN:   -debug-only=function-import \
; RUN:   -import-instr-limit=7 \
; RUN:   -import-declaration \
; RUN:   -thinlto-distributed-indexes \
; RUN:   -r=main.bc,main,px \
; RUN:   -r=main.bc,small_func, \
; RUN:   -r=main.bc,large_func, \
; RUN:   -r=lib.bc,callee,pl \
; RUN:   -r=lib.bc,large_indirect_callee,px \
; RUN:   -r=lib.bc,small_func,px \
; RUN:   -r=lib.bc,large_func,px \
; RUN:   -r=lib.bc,large_indirect_callee_alias,px \
; RUN:   -r=lib.bc,calleeAddrs,px -o summary main.bc lib.bc 2>&1 | FileCheck %s --check-prefix=DUMP
;
; RUN: llvm-lto -thinlto-action=thinlink -import-declaration -import-instr-limit=7  -o combined.index.bc main.bc lib.bc
; RUN: llvm-lto -thinlto-action=distributedindexes -debug-only=function-import -import-declaration -import-instr-limit=7 -thinlto-index combined.index.bc main.bc lib.bc 2>&1 | FileCheck %s --check-prefix=DUMP
; RUN: llvm-dis main.bc.thinlto.bc -o - | FileCheck %s --check-prefix=MAIN-DIS

; DUMP: - 2 function definitions and 3 function declarations imported from lib.bc

; main.ll should import {large_func, large_indirect_callee} as declarations.
; 
; First disassemble per-module summary and find out the GUID for {large_func, large_indirect_callee}.
;
; RUN: llvm-dis lib.bc -o - | FileCheck %s --check-prefix=LIB-DIS
; LIB-DIS: [[LIBMOD:\^[0-9]+]] = module: (path: "lib.bc", hash: (0, 0, 0, 0, 0))
; LIB-DIS: [[LARGEFUNC:\^[0-9]+]] = gv: (name: "large_func", summaries: {{.*}}) ; guid = 2418497564662708935
; LIB-DIS: [[LARGEINDIRECT:\^[0-9]+]] = gv: (name: "large_indirect_callee", summaries: {{.*}}) ; guid = 14343440786664691134
;
; Secondly disassemble main's combined summary and verify the import type of
; these two GUIDs are declaration.
;
; RUN: llvm-dis main.bc.thinlto.bc -o - | FileCheck %s --check-prefix=MAIN-DIS
;
; MAIN-DIS: [[LIBMOD:\^[0-9]+]] = module: (path: "lib.bc", hash: (0, 0, 0, 0, 0))
; MAIN-DIS: [[LARGEFUNC:\^[0-9]+]] = gv: (guid: 2418497564662708935, summaries: (function: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), insts: 8, {{.*}})))
; MAIN-DIS: [[LARGEINDIRECT:\^[0-9]+]] = gv: (guid: 14343440786664691134, summaries: (function: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), insts: 8, {{.*}})))
; MAIN-DIS: [[LARGEINDIRECTALIAS:\^[0-9]+]] = gv: (guid: 16730173943625350469, summaries: (alias: (module: [[LIBMOD]], flags: ({{.*}} importType: declaration), aliasee: [[LARGEINDIRECT]])))

; TODO: add test coverage for lto indexing stats (-debug-only=function-import, and 'reqasserts').

; Run in-process ThinLTO and tests that `callee` remains internalized even if
; the symbols of its callers (large_func and large_indirect_callee) are exported
; and visible to main module.

; RUN: llvm-lto2 run \
; RUN:   -save-temps \
; RUN:   -import-instr-limit=7 \
; RUN:   -import-declaration \
; RUN:   -r=main.bc,main,px \
; RUN:   -r=main.bc,small_func, \
; RUN:   -r=main.bc,large_func, \
; RUN:   -r=lib.bc,callee,pl \
; RUN:   -r=lib.bc,large_indirect_callee,px \
; RUN:   -r=lib.bc,small_func,px \
; RUN:   -r=lib.bc,large_func,px \
; RUN:   -r=lib.bc,large_indirect_callee_alias,px \
; RUN:   -r=lib.bc,calleeAddrs,px -o in-process main.bc lib.bc

; RUN: llvm-dis in-process.2.2.internalize.bc -o - | FileCheck %s --check-prefix=INTERNALIZE
; INTERNALIZE: define internal void @callee()

;--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
  call void @small_func()
  call void @large_func()
  ret i32 0
}

declare void @small_func()

; large_func without attributes
declare void @large_func()

;--- lib.ll
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@calleeAddrs = global [3 x ptr] [ptr @large_indirect_callee, ptr @small_indirect_callee, ptr @large_indirect_callee_alias]

define void @callee() #1 {
  ret void
}

define void @large_indirect_callee()#2 {
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  ret void
}

define internal void @small_indirect_callee() #0 {
  ret void
}

@large_indirect_callee_alias = alias void(), ptr @large_indirect_callee

define void @small_func() {
entry:
  %0 = load ptr, ptr @calleeAddrs
  call void %0(), !prof !0
  %1 = load ptr, ptr getelementptr inbounds ([3 x ptr], ptr @calleeAddrs, i64 0, i64 1)
  call void %1(), !prof !1
  %2 = load ptr, ptr getelementptr inbounds ([3 x ptr], ptr @calleeAddrs, i64 0, i64 2)
  call void %2(), !prof !2
  ret void
}

define void @large_func() #0 {
entry:
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  call void @callee()
  ret void
}

attributes #0 = { nounwind norecurse }

attributes #1 = { noinline }

attributes #2 = { norecurse }

!0 = !{!"VP", i32 0, i64 1, i64 14343440786664691134, i64 1}
!1 = !{!"VP", i32 0, i64 1, i64 13568239288960714650, i64 1}
!2 = !{!"VP", i32 0, i64 1, i64 16730173943625350469, i64 1}
