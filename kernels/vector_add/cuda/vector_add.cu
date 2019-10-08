
// CUDA version of vector add example

#include <stdio.h>

#define RealType float

__global__ void vector_add(const RealType *a, const RealType *b, RealType *c, const int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

void check_status(cudaError_t status, const char *api_name)
{
  if (status != cudaSuccess) {
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

  cudaError_t da_err = cudaMalloc(&device_a, bytes);
  check_status(da_err,"cudaMalloc for a");

  cudaError_t db_err = cudaMalloc(&device_b, bytes);
  check_status(db_err,"cudaMalloc for b");

  cudaError_t dc_err = cudaMalloc(&device_c, bytes);
  check_status(dc_err,"cudaMalloc for c");

  for (int i = 0; i < N; i++) {
    host_a[i] = 1.0;
    host_b[i] = 1.0 * i;
  }

  cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int blockSize = 1024;
  int gridSize = (int)(double(N)/blockSize) + 1;

  cudaEventRecord(start);
  vector_add<<<gridSize, blockSize>>>(device_a, device_b, device_c, N);
  cudaEventRecord(stop);

  cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);


  printf("c[0] = %g\n",host_c[0]);
  printf("c[N-1] = %g\n",host_c[N-1]);

  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  printf("kernel (ms) = %g\n",kernel_ms);

  double bw = 3*bytes*1e-6/kernel_ms;
  printf("BW (GB/s) = %g\n",bw);


  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  delete[] host_a;
  delete[] host_b;
  delete[] host_c;

  return 0;
}
