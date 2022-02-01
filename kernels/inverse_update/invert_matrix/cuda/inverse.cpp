
// Compute matrix inverse 

#include <vector>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void init_A(int n, std::vector<double>& A)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i*n + j] = i+j;
    }
  }
  for (int i = 0; i < n; i++) {
    A[i*n + i] += 10.0;
  }
}

void init_identity(int n, std::vector<double>& I)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      I[i*n + j] = 0.0;
    }
  }
  for (int i = 0; i < n; i++) {
    I[i*n + i] += 1.0;
  }
}

int main()
{
  const int N = 3;

  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat = cudaSuccess;

  cusolverDnHandle_t cusolverH = nullptr;

  status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  int lwork = 0;

  std::vector<double> A(N*N);
  init_A(N,A);

  std::vector<double> B(N*N);
  init_identity(N,B);

 
  double *d_A = nullptr; // device copy of A
  cudaStat = cudaMalloc((void**)&d_A, sizeof(double)*N*N);
  assert(cudaSuccess == cudaStat);

  double *d_B = nullptr; // device copy of B
  cudaStat = cudaMalloc((void**)&d_B, sizeof(double)*N*N);
  assert(cudaSuccess == cudaStat);

  int *d_ipiv = nullptr;
  cudaStat = cudaMalloc((void **)&d_ipiv, sizeof(int)*N);
  assert(cudaSuccess == cudaStat);

  int *d_info = nullptr;
  cudaStat = cudaMalloc((void **)&d_info, sizeof(int));
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMemcpy(d_A, A.data(), sizeof(double)*N*N, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMemcpy(d_B, B.data(), sizeof(double)*N*N, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat);



  status = cusolverDnDgetrf_bufferSize(
      cusolverH,
      N,
      N,
      d_A,
      N,
      &lwork);

  assert(CUSOLVER_STATUS_SUCCESS == status);

  std::cout <<"lwork = " << lwork << std::endl;

  double* d_work = nullptr;
  cudaStat = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  assert(cudaSuccess == cudaStat);


  status = cusolverDnDgetrf(
            cusolverH,
            N,
            N,
            d_A,
            N,
            d_work,
            d_ipiv,
            d_info);

  assert(CUSOLVER_STATUS_SUCCESS == status);

  int info;
  cudaStat = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat);

  std::cout << " info = " << info << std::endl;



  status = cusolverDnDgetrs(
            cusolverH,
            CUBLAS_OP_N,
            N,
            N,
            d_A,
            N,
            d_ipiv,
            d_B,
            N,
            d_info);
  assert(cudaSuccess == cudaStat);

  cudaStat = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat);

  std::cout << " info = " << info << std::endl;

}

