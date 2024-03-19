; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@var = global i32 0

; CHECK: @auth_var = global ptr ptrauth (ptr @var, i32 0, ptr null, i64 -1)
@auth_var = global ptr ptrauth (ptr @var, i32 0, ptr null, i64 -1)


; CHECK: @addrdisc_var = global ptr ptrauth (ptr @var, i32 0, ptr @addrdisc_var, i64 1234)
@addrdisc_var = global ptr ptrauth (ptr @var, i32 0, ptr @addrdisc_var, i64 1234)

; CHECK: @keyed_var = global ptr ptrauth (ptr @var, i32 3, ptr null, i64 0)
@keyed_var = global ptr ptrauth (ptr @var, i32 3, ptr null, i64 0)
