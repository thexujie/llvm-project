; RUN: llc -mtriple=aarch64-linux-gnu -global-isel=1 < %s | FileCheck %s

; CHECK-NOT: movk

define dso_local noundef i1 @ule_11111111(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 286331154
  ret i1 %2
}

define dso_local noundef i1 @ule_22222222(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 572662307
  ret i1 %2
}

define dso_local noundef i1 @ule_33333333(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 858993460
  ret i1 %2
}

define dso_local noundef i1 @ule_44444444(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 1145324613
  ret i1 %2
}

define dso_local noundef i1 @ule_55555555(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 1431655766
  ret i1 %2
}

define dso_local noundef i1 @ule_66666666(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 1717986919
  ret i1 %2
}

define dso_local noundef i1 @ule_77777777(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, 2004318072
  ret i1 %2
}

define dso_local noundef i1 @ule_88888888(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, -2004318071
  ret i1 %2
}

define dso_local noundef i1 @ule_99999999(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, -1717986918
  ret i1 %2
}

define dso_local noundef i1 @uge_11111111(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 286331152
  ret i1 %2
}

define dso_local noundef i1 @uge_22222222(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 572662305
  ret i1 %2
}

define dso_local noundef i1 @uge_33333333(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 858993458
  ret i1 %2
}

define dso_local noundef i1 @uge_44444444(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 1145324611
  ret i1 %2
}

define dso_local noundef i1 @uge_55555555(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 1431655764
  ret i1 %2
}

define dso_local noundef i1 @uge_66666666(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 1717986917
  ret i1 %2
}

define dso_local noundef i1 @uge_77777777(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 2004318070
  ret i1 %2
}

define dso_local noundef i1 @uge_88888888(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, -2004318073
  ret i1 %2
}

define dso_local noundef i1 @uge_99999999(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, -1717986920
  ret i1 %2
}

define dso_local noundef i1 @sle_11111111(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 286331154
  ret i1 %2
}

define dso_local noundef i1 @sle_22222222(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 572662307
  ret i1 %2
}

define dso_local noundef i1 @sle_33333333(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 858993460
  ret i1 %2
}

define dso_local noundef i1 @sle_44444444(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1145324613
  ret i1 %2
}

define dso_local noundef i1 @sle_55555555(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1431655766
  ret i1 %2
}

define dso_local noundef i1 @sle_66666666(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1717986919
  ret i1 %2
}

define dso_local noundef i1 @sle_77777777(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 2004318072
  ret i1 %2
}

define dso_local noundef i1 @sle_88888888(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, -2004318071
  ret i1 %2
}

define dso_local noundef i1 @sle_99999999(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i32 %0, -1717986918
  ret i1 %2
}

define dso_local noundef i1 @sge_11111111(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 286331152
  ret i1 %2
}

define dso_local noundef i1 @sge_22222222(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 572662305
  ret i1 %2
}

define dso_local noundef i1 @sge_33333333(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 858993458
  ret i1 %2
}

define dso_local noundef i1 @sge_44444444(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 1145324611
  ret i1 %2
}

define dso_local noundef i1 @sge_55555555(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 1431655764
  ret i1 %2
}

define dso_local noundef i1 @sge_66666666(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 1717986917
  ret i1 %2
}

define dso_local noundef i1 @sge_77777777(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 2004318070
  ret i1 %2
}

define dso_local noundef i1 @sge_88888888(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, -2004318073
  ret i1 %2
}

define dso_local noundef i1 @sge_99999999(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, -1717986920
  ret i1 %2
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }