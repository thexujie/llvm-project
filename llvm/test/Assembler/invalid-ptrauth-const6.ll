; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: signed pointer address discriminator must be a pointer
@auth_var = global ptr ptrauth (ptr @var, i32 2, i8 0, i32 1000000)
