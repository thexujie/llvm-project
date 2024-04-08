//=== ifunc_select.c - Check environment hardware feature -*- C -*-===========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__linux__)

static long syscall_impl_5_args(long number, long arg1, long arg2, long arg3,
                                long arg4, long arg5) {
  register long a7 __asm__("a7") = number;
  register long a0 __asm__("a0") = arg1;
  register long a1 __asm__("a1") = arg2;
  register long a2 __asm__("a2") = arg3;
  register long a3 __asm__("a3") = arg4;
  register long a4 __asm__("a4") = arg5;
  __asm__ __volatile__("ecall\n\t"
                       : "=r"(a0)
                       : "r"(a7), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4)
                       : "memory");
  return a0;
}

struct riscv_hwprobe {
  long long key;
  unsigned long long value;
};

// Note: sync with https://docs.kernel.org/arch/riscv/hwprobe.html
#define RISCV_HWPROBE_KEY_MVENDORID 0
#define RISCV_HWPROBE_KEY_MARCHID 1
#define RISCV_HWPROBE_KEY_MIMPID 2
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_BASE_BEHAVIOR_IMA (1 << 0)
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define RISCV_HWPROBE_IMA_FD (1 << 0)
#define RISCV_HWPROBE_IMA_C (1 << 1)
#define RISCV_HWPROBE_IMA_V (1 << 2)
#define RISCV_HWPROBE_EXT_ZBA (1 << 3)
#define RISCV_HWPROBE_EXT_ZBB (1 << 4)
#define RISCV_HWPROBE_EXT_ZBS (1 << 5)
#define RISCV_HWPROBE_EXT_ZICBOZ (1 << 6)
#define RISCV_HWPROBE_EXT_ZBC (1 << 7)
#define RISCV_HWPROBE_EXT_ZBKB (1 << 8)
#define RISCV_HWPROBE_EXT_ZBKC (1 << 9)
#define RISCV_HWPROBE_EXT_ZBKX (1 << 10)
#define RISCV_HWPROBE_EXT_ZKND (1 << 11)
#define RISCV_HWPROBE_EXT_ZKNE (1 << 12)
#define RISCV_HWPROBE_EXT_ZKNH (1 << 13)
#define RISCV_HWPROBE_EXT_ZKSED (1 << 14)
#define RISCV_HWPROBE_EXT_ZKSH (1 << 15)
#define RISCV_HWPROBE_EXT_ZKT (1 << 16)
#define RISCV_HWPROBE_EXT_ZVBB (1 << 17)
#define RISCV_HWPROBE_EXT_ZVBC (1 << 18)
#define RISCV_HWPROBE_EXT_ZVKB (1 << 19)
#define RISCV_HWPROBE_EXT_ZVKG (1 << 20)
#define RISCV_HWPROBE_EXT_ZVKNED (1 << 21)
#define RISCV_HWPROBE_EXT_ZVKNHA (1 << 22)
#define RISCV_HWPROBE_EXT_ZVKNHB (1 << 23)
#define RISCV_HWPROBE_EXT_ZVKSED (1 << 24)
#define RISCV_HWPROBE_EXT_ZVKSH (1 << 25)
#define RISCV_HWPROBE_EXT_ZVKT (1 << 26)
#define RISCV_HWPROBE_EXT_ZFH (1 << 27)
#define RISCV_HWPROBE_EXT_ZFHMIN (1 << 28)
#define RISCV_HWPROBE_EXT_ZIHINTNTL (1 << 29)
#define RISCV_HWPROBE_EXT_ZVFH (1 << 30)
#define RISCV_HWPROBE_EXT_ZVFHMIN (1 << 31)
#define RISCV_HWPROBE_EXT_ZFA (1ULL << 32)
#define RISCV_HWPROBE_EXT_ZTSO (1ULL << 33)
#define RISCV_HWPROBE_EXT_ZACAS (1ULL << 34)
#define RISCV_HWPROBE_EXT_ZICOND (1ULL << 35)
#define RISCV_HWPROBE_KEY_CPUPERF_0 5
#define RISCV_HWPROBE_MISALIGNED_UNKNOWN (0 << 0)
#define RISCV_HWPROBE_MISALIGNED_EMULATED (1 << 0)
#define RISCV_HWPROBE_MISALIGNED_SLOW (2 << 0)
#define RISCV_HWPROBE_MISALIGNED_FAST (3 << 0)
#define RISCV_HWPROBE_MISALIGNED_UNSUPPORTED (4 << 0)
#define RISCV_HWPROBE_MISALIGNED_MASK (7 << 0)
#define RISCV_HWPROBE_KEY_ZICBOZ_BLOCK_SIZE 6
/* Increase RISCV_HWPROBE_MAX_KEY when adding items. */

/* Flags */
#define RISCV_HWPROBE_WHICH_CPUS (1 << 0)

/* Size definition for CPU sets.  */
#define __CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long int))

/* Data structure to describe CPU mask.  */
typedef struct {
  unsigned long int __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define SYS_riscv_hwprobe 258
static long sys_riscv_hwprobe(struct riscv_hwprobe *pairs, unsigned pair_count,
                              unsigned cpu_count, cpu_set_t *cpus,
                              unsigned int flags) {
  return syscall_impl_5_args(SYS_riscv_hwprobe, (long)pairs, pair_count,
                             cpu_count, (long)cpus, flags);
}

static long initHwProbe(struct riscv_hwprobe *Hwprobes, int len) {
  return sys_riscv_hwprobe(Hwprobes, len, 0, (cpu_set_t *)((void *)0), 0);
}

#endif // defined(__linux__)

unsigned __riscv_ifunc_select(struct riscv_hwprobe *ReqireKeys,
                              unsigned Length) {
#if defined(__linux__)
  // Init Hwprobe
  struct riscv_hwprobe Pairs[64];

  for (unsigned Idx = 0; Idx < Length; Idx++) {
    Pairs[Idx].key = ReqireKeys[Idx].key;
    Pairs[Idx].value = 0;
  }

  // hwprobe not success
  if (initHwProbe(Pairs, 2))
    return 0;

  for (unsigned Idx = 0; Idx < Length; Idx++) {
    if (Pairs[Idx].key == -1)
      return 0;

    if ((ReqireKeys[Idx].value & Pairs[Idx].value) != ReqireKeys[Idx].value)
      return 0;
  }

  return 1;
#else
  // If other platform support IFUNC, need to implement its
  // __riscv_ifunc_select.
  return 0;
#endif
}
