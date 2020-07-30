
// Test to see if OpenMP code is really running on target device

#include <omp.h>
#include <stdio.h>

int main() {
  int isHost = -1;

#pragma omp target map(from : isHost)
  { isHost = omp_is_initial_device(); }

  if (isHost < 0) {
    printf("Runtime error, isHost = %d\n", isHost);
  } else {
    printf("Offload okay, isHost = %d\n", isHost);
  }
  return 0;
}
