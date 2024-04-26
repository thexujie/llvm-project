; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -mtriple riscv64-linux-gnu -mattr=+v,+f -S -disable-output -debug-only=loop-vectorize 2>&1 | FileCheck %s

; CHECK: LV: Adding cost of generating tail-fold mask for VF 1: 0
; CHECK: LV: Adding cost of generating tail-fold mask for VF 2: 2
; CHECK: LV: Adding cost of generating tail-fold mask for VF 4: 4
; CHECK: LV: Adding cost of generating tail-fold mask for VF 8: 8
; CHECK: LV: Adding cost of generating tail-fold mask for VF vscale x 1: 2
; CHECK: LV: Adding cost of generating tail-fold mask for VF vscale x 2: 4
; CHECK: LV: Adding cost of generating tail-fold mask for VF vscale x 4: 8

define void @simple_memset(i32 %val, ptr %ptr, i64 %n) #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep = getelementptr i32, ptr %ptr, i64 %index
  store i32 %val, ptr %gep
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  ret void
}
