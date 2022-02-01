
// Compute matrix inverse as a first step

#include <vector>
#include <iostream>
#include <assert.h>
#include "magma_v2.h"
#include "magma_lapack.h"

void init_A(int n, std::vector<double>& A)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i*n + j] = i+2*j;
    } }
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

void print_matrix(int n, std::vector<double>& M)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << M[i*n + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main()
{
  magma_init();
  const int N = 3;

  magma_queue_t queue=NULL;
  magma_int_t dev = 0;
  magma_queue_create(dev, &queue);

  std::vector<double> A(N*N);
  init_A(N,A);
  print_matrix(N,A);

  double* d_A;
  magma_int_t ret = magma_dmalloc(&d_A, N*N);
  if (ret != MAGMA_SUCCESS) {
    std::cout << "magma dmalloc failed\n";
  }
  magma_dsetmatrix(N, N, A.data(), N, d_A, N, queue);

  std::vector<double> B(N*N);
  init_identity(N,B);

  int info;
  std::vector<int> ipiv(N);

#if 0
  int status = magma_dgetrf(
      N,
      N,
      A.data(),
      N,
      ipiv.data(),
      &info
      );
#endif
  int* d_ipiv;
  ret = magma_imalloc(&d_ipiv, N);
  if (ret != MAGMA_SUCCESS) {
    std::cout << "magma imalloc failed\n";
  }
  magma_isetmatrix(N, N, ipiv.data(), N, d_ipiv, N, queue);

  int status = magma_dgetrf_gpu(
      N,
      N,
      d_A,
      N,
      ipiv.data(),
      &info
      );

  std::cout << " status from dgetrf = " << status << std::endl;
  std::cout << " info from degetrf = " << info << std::endl;
  magma_dgetmatrix(N, N, d_A, N, A.data(), N, queue);

  print_matrix(N,A);

  int nb = magma_get_dgetri_nb(N);
  int lwork = nb*N;
  std::cout << " lwork = " << lwork << std::endl;

  std::vector<double> dwork(lwork);

  double* d_work;
  ret = magma_dmalloc(&d_work, lwork);
  if (ret != MAGMA_SUCCESS) {
    std::cout << "magma dmalloc for dwork failed\n";
  }
  magma_dsetvector(lwork, dwork.data(), 1, d_work, 1, queue);

  status = magma_dgetri_gpu(
      N,
      d_A,
      N,
      ipiv.data(),
      d_work,
      lwork,
      &info);


  std::cout << " status from dgetri = " << status << std::endl;
  std::cout << " info from degetri = " << info << std::endl;
  magma_dgetmatrix(N, N, d_A, N, A.data(), N, queue);

  print_matrix(N,A);

  magma_queue_destroy(queue);
  magma_finalize();
  return 0;
}

