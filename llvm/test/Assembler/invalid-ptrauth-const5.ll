; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: signed pointer discriminator must be i64 constant integer
@auth_var = global ptr ptrauth (ptr @var, i32 2, ptr null, i64 ptrtoint (ptr @var to i64))
