// RUN: %libomptarget-compilexx-run-and-check-generic
//
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <assert.h>
#include <ompx.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  const int num_blocks = 1;
  const int block_size = 64;
  const int N = num_blocks * block_size;
  unsigned *data = (int *)malloc(N * sizeof(unsigned));

  for (int i = 0; i < N; ++i)
    data[i] = i & 0x1;

#pragma omp target teams ompx_bare num_teams(num_blocks) thread_limit(block_size) map(tofrom: data[0:N])
  {
    int tid = ompx_thread_id_x();
    unsigned mask = ompx_ballot_sync(0xffffffff, data[tid]);
    data[tid] += mask;
  }

  for (int i = 0; i < N; ++i)
    assert(data[i] == ((i & 0x1) + 0xaaaaaaaa));

  // CHECK: PASS
  printf("PASS\n");

  return 0;
}
