
// Compute matrix inverse

#include <vector>
#include <iostream>
#include <assert.h>

#define dgetrf dgetrf_
#define dgetri dgetri_

extern "C" {
void dgetrf(const int* n,
            const int* m,
            double* a,
            const int* lda,
            int* piv,
            int* info);

void dgetri(const int* n,
             double* a,
             const int* lda,
             int const* piv,
             double* work,
             const int* lwork,
             int* info);
}


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
  const int N = 3;

  std::vector<double> A(N*N);
  init_A(N,A);
  print_matrix(N,A);

  std::vector<double> B(N*N);
  init_identity(N,B);

  int info;
  std::vector<int> ipiv(N);

  dgetrf(&N, &N, A.data(), &N, ipiv.data(), &info);
  if (info != 0) {
    std::cout << "dgetrf error, info = " << info << std::endl;
  }

  print_matrix(N,A);

  int lwork = -1;
  std::vector<double> work(1);

  dgetri(&N, A.data(), &N, ipiv.data(), work.data(), &lwork, &info);

  lwork = work[0];
  work.resize(lwork);

  dgetri(&N, A.data(), &N, ipiv.data(), work.data(), &lwork, &info);

  if (info != 0) {
    std::cout << "dgetri error, info = " << info << std::endl;
  }
  print_matrix(N,A);

  return 0;
}

