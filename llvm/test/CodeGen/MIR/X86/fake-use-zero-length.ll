; RUN: llc < %s -stop-after=finalize-isel | FileCheck %s --implicit-check-not=FAKE_USE
;
; Make sure SelectionDAG does not crash handling fake uses of zero-length arrays
; and structs. Check also that they are not propagated.
;
; Generated from the following source with
; clang -fextend-lifetimes -S -emit-llvm -O2 -mllvm -stop-after=safe-stack -o test.mir test.cpp
;
; int main ()
; { int array[0]; }
;
;
; CHECK: liveins: $[[IN_REG:[a-zA-Z0-9]+]]
; CHECK: %[[IN_VREG:[a-zA-Z0-9]+]]:gr32 = COPY $[[IN_REG]]
; CHECK: FAKE_USE %[[IN_VREG]]

source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define hidden i32 @main([0 x i32] %zero, [1 x i32] %one) local_unnamed_addr {
entry:
  notail call void (...) @bar([0 x i32] %zero)
  notail call void (...) @baz([1 x i32] %one)
  notail call void (...) @llvm.fake.use([0 x i32] %zero)
  notail call void (...) @llvm.fake.use([1 x i32] %one)
  ret i32 0
}

declare void @bar([0 x i32] %a)
declare void @baz([1 x i32] %a)

; Function Attrs: nounwind
declare void @llvm.fake.use(...)

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 10.0.0"}