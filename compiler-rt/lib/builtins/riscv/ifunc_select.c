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

unsigned __riscv_ifunc_select(struct riscv_hwprobe *RequireKeys,
                              unsigned Length) {
#if defined(__linux__)
  // Init Hwprobe
  struct riscv_hwprobe HwprobePairs[Length];

  for (unsigned Idx = 0; Idx < Length; Idx++) {
    HwprobePairs[Idx].key = RequireKeys[Idx].key;
    HwprobePairs[Idx].value = 0;
  }

  // hwprobe not success
  if (initHwProbe(HwprobePairs, Length))
    return 0;

  for (unsigned Idx = 0; Idx < Length; Idx++) {
    if (HwprobePairs[Idx].key == -1)
      return 0;

    if ((RequireKeys[Idx].value & HwprobePairs[Idx].value) !=
        RequireKeys[Idx].value)
      return 0;
  }

  return 1;
#else
  // If other platform support IFUNC, need to implement its
  // __riscv_ifunc_select.
  return 0;
#endif
}
