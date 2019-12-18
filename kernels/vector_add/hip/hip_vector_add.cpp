#include "hip/hip_runtime.h"

// HIP-ified version of vector add example using hipify-perl

#include <stdio.h>

#define RealType float

__global__ void vector_add(const RealType *a, const RealType *b, RealType *c, const int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

void check_status(hipError_t status, const char *api_name)
{
  if (status != hipSuccess) {
    printf("%s failed\n",api_name);
  }
}


int main()
{
  const int N = 1024*1024;

  RealType *host_a;
  RealType *host_b;
  RealType *host_c;

  RealType *device_a;
  RealType *device_b;
  RealType *device_c;

  size_t bytes = N*sizeof(RealType);
  double array_mb = bytes/1024.0/1024.0;
  printf("Array size (MiB) = %g\n",array_mb);

  host_a = new RealType[N];
  host_b = new RealType[N];
  host_c = new RealType[N];

  hipError_t da_err = hipMalloc(&device_a, bytes);
  check_status(da_err,"hipMalloc for a");

  hipError_t db_err = hipMalloc(&device_b, bytes);
  check_status(db_err,"hipMalloc for b");

  hipError_t dc_err = hipMalloc(&device_c, bytes);
  check_status(dc_err,"hipMalloc for c");

  for (int i = 0; i < N; i++) {
    host_a[i] = 1.0;
    host_b[i] = 1.0 * i;
  }

  hipMemcpy(device_a, host_a, bytes, hipMemcpyHostToDevice);
  hipMemcpy(device_b, host_b, bytes, hipMemcpyHostToDevice);

  hipEvent_t start;
  hipEvent_t stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  int blockSize = 1024;
  int gridSize = (int)(double(N)/blockSize) + 1;

  hipEventRecord(start);
  hipLaunchKernelGGL((vector_add), dim3(gridSize), dim3(blockSize), 0, 0, device_a, device_b, device_c, N);
  hipEventRecord(stop);

  hipMemcpy(host_c, device_c, bytes, hipMemcpyDeviceToHost);
  hipEventSynchronize(stop);


  printf("c[0] = %g\n",host_c[0]);
  printf("c[N-1] = %g\n",host_c[N-1]);

  float kernel_ms;
  hipEventElapsedTime(&kernel_ms, start, stop);
  printf("kernel (ms) = %g\n",kernel_ms);

  double bw = 3*bytes*1e-6/kernel_ms;
  printf("BW (GB/s) = %g\n",bw);


  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipFree(device_a);
  hipFree(device_b);
  hipFree(device_c);

  delete[] host_a;
  delete[] host_b;
  delete[] host_c;

  return 0;
}
