// RUN: %clang_cc1 -std=c++11 -triple riscv64-linux-gnu -target-feature +i -target-feature +m -emit-llvm %s -o - | FileCheck %s

__attribute__((target_clones("default", "arch=rv64gc"))) int foo1(void) { return 1; }
__attribute__((target_clones("default", "arch=+zbb"))) int foo2(void) { return 2; }
__attribute__((target_clones("default", "arch=+zbb,+c"))) int foo3(void) { return 3; }
__attribute__((target_clones("default", "arch=rv64gc", "arch=+zbb,+v"))) int foo4(void) { return 4; }
__attribute__((target_clones("default"))) int foo5(void) { return 5; }

int bar() {
  return foo1() + foo2() + foo3() + foo4();
}


//.
// CHECK: @[[GLOB0:[0-9]+]] = private unnamed_addr constant [25 x i8] c"m;a;f;d;c;zicsr;zifencei\00", align 1
// CHECK: @[[GLOB1:[0-9]+]] = private unnamed_addr constant [8 x i8] c"i;m;zbb\00", align 1
// CHECK: @[[GLOB2:[0-9]+]] = private unnamed_addr constant [10 x i8] c"i;m;zbb;c\00", align 1
// CHECK: @[[GLOB3:[0-9]+]] = private unnamed_addr constant [25 x i8] c"m;a;f;d;c;zicsr;zifencei\00", align 1
// CHECK: @[[GLOB4:[0-9]+]] = private unnamed_addr constant [10 x i8] c"i;m;zbb;v\00", align 1
// CHECK: @_Z4foo1v.ifunc = weak_odr alias i32 (), ptr @_Z4foo1v
// CHECK: @_Z4foo2v.ifunc = weak_odr alias i32 (), ptr @_Z4foo2v
// CHECK: @_Z4foo3v.ifunc = weak_odr alias i32 (), ptr @_Z4foo3v
// CHECK: @_Z4foo4v.ifunc = weak_odr alias i32 (), ptr @_Z4foo4v
// CHECK: @_Z4foo5v.ifunc = weak_odr alias i32 (), ptr @_Z4foo5v
// CHECK: @_Z4foo1v = weak_odr ifunc i32 (), ptr @_Z4foo1v.resolver
// CHECK: @_Z4foo2v = weak_odr ifunc i32 (), ptr @_Z4foo2v.resolver
// CHECK: @_Z4foo3v = weak_odr ifunc i32 (), ptr @_Z4foo3v.resolver
// CHECK: @_Z4foo4v = weak_odr ifunc i32 (), ptr @_Z4foo4v.resolver
// CHECK: @_Z4foo5v = weak_odr ifunc i32 (), ptr @_Z4foo5v.resolver
//.
// CHECK-LABEL: define dso_local noundef signext i32 @_Z4foo1v.default(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 1
//
//
// CHECK-LABEL: define weak_odr ptr @_Z4foo1v.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call i1 @__riscv_ifunc_select(ptr @[[GLOB0]])
// CHECK-NEXT:    br i1 [[TMP0]], label [[RESOLVER_RETURN:%.*]], label [[RESOLVER_ELSE:%.*]]
// CHECK:       resolver_return:
// CHECK-NEXT:    ret ptr @"_Z4foo1v.arch=rv64gc"
// CHECK:       resolver_else:
// CHECK-NEXT:    ret ptr @_Z4foo1v.default
//
//
// CHECK-LABEL: define dso_local noundef signext i32 @_Z4foo2v.default(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 2
//
//
// CHECK-LABEL: define weak_odr ptr @_Z4foo2v.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call i1 @__riscv_ifunc_select(ptr @[[GLOB1]])
// CHECK-NEXT:    br i1 [[TMP0]], label [[RESOLVER_RETURN:%.*]], label [[RESOLVER_ELSE:%.*]]
// CHECK:       resolver_return:
// CHECK-NEXT:    ret ptr @"_Z4foo2v.arch=+zbb"
// CHECK:       resolver_else:
// CHECK-NEXT:    ret ptr @_Z4foo2v.default
//
//
// CHECK-LABEL: define dso_local noundef signext i32 @_Z4foo3v.default(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 3
//
//
// CHECK-LABEL: define weak_odr ptr @_Z4foo3v.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call i1 @__riscv_ifunc_select(ptr @[[GLOB2]])
// CHECK-NEXT:    br i1 [[TMP0]], label [[RESOLVER_RETURN:%.*]], label [[RESOLVER_ELSE:%.*]]
// CHECK:       resolver_return:
// CHECK-NEXT:    ret ptr @"_Z4foo3v.arch=+zbb,+c"
// CHECK:       resolver_else:
// CHECK-NEXT:    ret ptr @_Z4foo3v.default
//
//
// CHECK-LABEL: define dso_local noundef signext i32 @_Z4foo4v.default(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 4
//
//
// CHECK-LABEL: define weak_odr ptr @_Z4foo4v.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call i1 @__riscv_ifunc_select(ptr @[[GLOB3]])
// CHECK-NEXT:    br i1 [[TMP0]], label [[RESOLVER_RETURN:%.*]], label [[RESOLVER_ELSE:%.*]]
// CHECK:       resolver_return:
// CHECK-NEXT:    ret ptr @"_Z4foo4v.arch=rv64gc"
// CHECK:       resolver_else:
// CHECK-NEXT:    [[TMP1:%.*]] = call i1 @__riscv_ifunc_select(ptr @[[GLOB4]])
// CHECK-NEXT:    br i1 [[TMP1]], label [[RESOLVER_RETURN1:%.*]], label [[RESOLVER_ELSE2:%.*]]
// CHECK:       resolver_return1:
// CHECK-NEXT:    ret ptr @"_Z4foo4v.arch=+zbb,+v"
// CHECK:       resolver_else2:
// CHECK-NEXT:    ret ptr @_Z4foo4v.default
//
//
// CHECK-LABEL: define dso_local noundef signext i32 @_Z4foo5v.default(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 5
//
//
// CHECK-LABEL: define weak_odr ptr @_Z4foo5v.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    ret ptr @_Z4foo5v.default
//
//
// CHECK-LABEL: define dso_local noundef signext i32 @_Z3barv(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call noundef signext i32 @_Z4foo1v()
// CHECK-NEXT:    [[CALL1:%.*]] = call noundef signext i32 @_Z4foo2v()
// CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[CALL]], [[CALL1]]
// CHECK-NEXT:    [[CALL2:%.*]] = call noundef signext i32 @_Z4foo3v()
// CHECK-NEXT:    [[ADD3:%.*]] = add nsw i32 [[ADD]], [[CALL2]]
// CHECK-NEXT:    [[CALL4:%.*]] = call noundef signext i32 @_Z4foo4v()
// CHECK-NEXT:    [[ADD5:%.*]] = add nsw i32 [[ADD3]], [[CALL4]]
// CHECK-NEXT:    ret i32 [[ADD5]]
//
//.
// CHECK: attributes #[[ATTR0]] = { mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i,+m" }
// CHECK: attributes #[[ATTR1:[0-9]+]] = { mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+a,+c,+d,+f,+m,+zicsr,+zifencei,-e,-experimental-smmpm,-experimental-smnpm,-experimental-ssnpm,-experimental-sspm,-experimental-ssqosid,-experimental-supm,-experimental-zaamo,-experimental-zabha,-experimental-zalasr,-experimental-zalrsc,-experimental-zcmop,-experimental-zfbfmin,-experimental-zicfilp,-experimental-zicfiss,-experimental-zimop,-experimental-ztso,-experimental-zvfbfmin,-experimental-zvfbfwma,-h,-shcounterenw,-shgatpa,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smepmp,-ssaia,-ssccptr,-sscofpmf,-sscounterenw,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-v,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xsfcease,-xsfvcp,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-za128rs,-za64rs,-zacas,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zca,-zcb,-zcd,-zce,-zcf,-zcmp,-zcmt,-zdinx,-zfa,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
// CHECK: attributes #[[ATTR2:[0-9]+]] = { mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i,+m,+zbb" }
// CHECK: attributes #[[ATTR3:[0-9]+]] = { mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+c,+i,+m,+zbb" }
// CHECK: attributes #[[ATTR4:[0-9]+]] = { mustprogress noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+d,+f,+i,+m,+v,+zbb,+zicsr,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b" }
//.
// CHECK: [[META0:![0-9]+]] = !{i32 1, !"wchar_size", i32 4}
// CHECK: [[META1:![0-9]+]] = !{i32 1, !"target-abi", !"lp64"}
// CHECK: [[META2:![0-9]+]] = !{i32 6, !"riscv-isa", [[META3:![0-9]+]]}
// CHECK: [[META3]] = !{!"rv64i2p1_m2p0"}
// CHECK: [[META4:![0-9]+]] = !{i32 8, !"SmallDataLimit", i32 0}
// CHECK: [[META5:![0-9]+]] = !{!"{{.*}}clang version {{.*}}"}
//.
