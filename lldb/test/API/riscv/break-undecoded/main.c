int main() {
  // This instruction is not valid, but we have an ability to set
  // software breakpoint.
  // This results illegal instruction during execution, not fail to set
  // breakpoint
  asm volatile(".insn r 0x73, 0, 0, a0, a1, a2" : :);
}
