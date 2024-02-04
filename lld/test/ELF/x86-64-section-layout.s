# REQUIRES: x86
## Test the placement of .lrodata, .lbss, .ldata, and their -fdata-sections variants.
## See also section-layout.s.

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 --defsym=BSS=1 a.s -o a.o
# RUN: ld.lld --section-start=.note=0x200300 a.o -o a
# RUN: llvm-readelf -S -l a | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a1.o
# RUN: ld.lld --section-start=.note=0x200300 a1.o -o a1
# RUN: llvm-readelf -S a1 | FileCheck %s --check-prefix=CHECK1

# RUN: ld.lld -T b.lds -z norelro a.o -o b
# RUN: llvm-readelf -S -l b | FileCheck %s --check-prefix=CHECK2

# CHECK:       Name       Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT:             NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT:  .note      NOTE            0000000000200300 000300 000001 00   A  0   0  1
# CHECK-NEXT:  .ltext     PROGBITS        0000000000201301 000301 000001 00 AXl  0   0  1
# CHECK-NEXT:  .lrodata   PROGBITS        0000000000202302 000302 000002 00  Al  0   0  1
# CHECK-NEXT:  .rodata    PROGBITS        0000000000202304 000304 000001 00   A  0   0  1
# CHECK-NEXT:  .text      PROGBITS        0000000000203308 000308 000001 00  AX  0   0  4
# CHECK-NEXT:  .tdata     PROGBITS        0000000000204309 000309 000001 00 WAT  0   0  1
# CHECK-NEXT:  .tbss      NOBITS          000000000020430a 00030a 000002 00 WAT  0   0  1
# CHECK-NEXT:  .relro_padding NOBITS      000000000020430a 00030a 000cf6 00  WA  0   0  1
# CHECK-NEXT:  .data      PROGBITS        000000000020530a 00030a 000001 00  WA  0   0  1
# CHECK-NEXT:  .bss       NOBITS          000000000020530b 00030b 001800 00  WA  0   0  1
## We spend size(.bss) % MAXPAGESIZE bytes for .bss.
# CHECK-NEXT:  .ldata     PROGBITS        0000000000207b0b 000b0b 000002 00 WAl  0   0  1
# CHECK-NEXT:  .ldata2    PROGBITS        0000000000207b0d 000b0d 000001 00 WAl  0   0  1
# CHECK-NEXT:  .lbss      NOBITS          0000000000207b0e 000b0e 000002 00 WAl  0   0  1
# CHECK-NEXT:  .comment   PROGBITS        0000000000000000 000b0e {{.*}} 01  MS  0   0  1

# CHECK:       Program Headers:
# CHECK-NEXT:    Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT:    PHDR  0x000040 0x0000000000200040 0x0000000000200040 {{.*}}   {{.*}}   R   0x8
# CHECK-NEXT:    LOAD  0x000000 0x0000000000200000 0x0000000000200000 0x000301 0x000301 R   0x1000
# CHECK-NEXT:    LOAD  0x000301 0x0000000000201301 0x0000000000201301 0x000001 0x000001 R E 0x1000
# CHECK-NEXT:    LOAD  0x000302 0x0000000000202302 0x0000000000202302 0x000003 0x000003 R   0x1000
# CHECK-NEXT:    LOAD  0x000308 0x0000000000203308 0x0000000000203308 0x000001 0x000001 R E 0x1000
# CHECK-NEXT:    LOAD  0x000309 0x0000000000204309 0x0000000000204309 0x000001 0x000cf7 RW  0x1000
# CHECK-NEXT:    LOAD  0x00030a 0x000000000020530a 0x000000000020530a 0x000001 0x001801 RW  0x1000
# CHECK-NEXT:    LOAD  0x000b0b 0x0000000000207b0b 0x0000000000207b0b 0x000003 0x000005 RW  0x1000

# CHECK1:      .data      PROGBITS        000000000020530a 00030a 000001 00  WA  0   0  1
# CHECK1-NEXT: .ldata     PROGBITS        000000000020530b 00030b 000002 00 WAl  0   0  1
# CHECK1-NEXT: .ldata2    PROGBITS        000000000020530d 00030d 000001 00 WAl  0   0  1
# CHECK1-NEXT: .comment   PROGBITS        0000000000000000 00030e {{.*}} 01  MS  0   0  1

# CHECK2:      .note      NOTE            0000000000200300 000300 000001 00   A  0   0  1
# CHECK2-NEXT: .ltext     PROGBITS        0000000000200301 000301 000001 00 AXl  0   0  1
# CHECK2-NEXT: .lrodata   PROGBITS        0000000000200302 000302 000001 00  Al  0   0  1
## With a SECTIONS command, we suppress the default rule placing .lrodata.* into .lrodata.
# CHECK2-NEXT: .lrodata.1 PROGBITS        0000000000200303 000303 000001 00  Al  0   0  1
# CHECK2-NEXT: .rodata    PROGBITS        0000000000200304 000304 000001 00   A  0   0  1
# CHECK2-NEXT: .text      PROGBITS        0000000000200308 000308 000001 00  AX  0   0  4
# CHECK2-NEXT: .tdata     PROGBITS        0000000000200309 000309 000001 00 WAT  0   0  1
# CHECK2-NEXT: .tbss      NOBITS          000000000020030a 00030a 000001 00 WAT  0   0  1
# CHECK2-NEXT: .tbss.1    NOBITS          000000000020030b 00030a 000001 00 WAT  0   0  1
# CHECK2-NEXT: .data      PROGBITS        000000000020030a 00030a 000001 00  WA  0   0  1
# CHECK2-NEXT: .bss       NOBITS          000000000020030b 00030b 001800 00  WA  0   0  1
# CHECK2-NEXT: .ldata     PROGBITS        0000000000201b0b 001b0b 000002 00 WAl  0   0  1
# CHECK2-NEXT: .ldata2    PROGBITS        0000000000201b0d 001b0d 000001 00 WAl  0   0  1
# CHECK2-NEXT: .lbss      NOBITS          0000000000201b0e 001b0e 000002 00 WAl  0   0  1
# CHECK2-NEXT: .comment   PROGBITS        0000000000000000 001b0e {{.*}} 01  MS  0   0  1

# CHECK2:      Program Headers:
# CHECK2-NEXT:   Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK2-NEXT:   PHDR  0x000040 0x0000000000200040 0x0000000000200040 {{.*}}   {{.*}}   R   0x8
# CHECK2-NEXT:   LOAD  0x000000 0x0000000000200000 0x0000000000200000 0x000301 0x000301 R   0x1000
# CHECK2-NEXT:   LOAD  0x000301 0x0000000000200301 0x0000000000200301 0x000001 0x000001 R E 0x1000
# CHECK2-NEXT:   LOAD  0x000302 0x0000000000200302 0x0000000000200302 0x000003 0x000003 R   0x1000
# CHECK2-NEXT:   LOAD  0x000308 0x0000000000200308 0x0000000000200308 0x000001 0x000001 R E 0x1000
# CHECK2-NEXT:   LOAD  0x000309 0x0000000000200309 0x0000000000200309 0x001805 0x001807 RW  0x1000
# CHECK2-NEXT:   TLS   0x000309 0x0000000000200309 0x0000000000200309 0x000001 0x000003 R   0x1

#--- a.s
.globl _start
_start:
  ret

.section .ltext,"axl",@progbits; .space 1
.section .note,"a",@note; .space 1
.section .rodata,"a",@progbits; .space 1
.section .data,"aw",@progbits; .space 1
.ifdef BSS
## .bss is large than one MAXPAGESIZE to test file offsets.
.section .bss,"aw",@nobits; .space 0x1800
.endif
.section .tdata,"awT",@progbits; .space 1
.section .tbss,"awT",@nobits; .space 1
.section .tbss.1,"awT",@nobits; .space 1

.section .lrodata,"al"; .space 1
.section .lrodata.1,"al"; .space 1
.section .ldata,"awl"; .space 1
## Input .ldata.rel.ro sections are placed in the output .ldata section.
.section .ldata.rel.ro,"awl"; .space 1
.ifdef BSS
.section .lbss,"awl",@nobits; .space 1
## Input .lbss.rel.ro sections are placed in the output .lbss section.
.section .lbss.rel.ro,"awl",@nobits; .space 1
.endif
.section .ldata2,"awl"; .space 1

#--- b.lds
SECTIONS {
  . = 0x200300;
  .rodata : {}
  .text : {}
  .data : {}
  .bss : {}
  .ldata : { *(.ldata .ldata.*) }
  .ldata2 : {}
  .lbss : { *(.lbss .lbss.*) }
}
