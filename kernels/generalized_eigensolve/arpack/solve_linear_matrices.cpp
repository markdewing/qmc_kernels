

#include <algorithm>
#include <arpack/arpack.hpp>
#include <chrono>
#include <cmath>
#include <hdf5.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// #define USE_MKL
#ifdef USE_MKL
#include <mkl.h>
#endif

#ifndef USE_MKL
extern "C" {
#define dgetrf dgetrf_
#define dgetri dgetri_
#define dgetrs dgetrs_
#define dgeev dgeev_
#define dggev dggev_
#define dgemm dgemm_
#define dgemv dgemv_
// void dgetrf(const int& n, const int& m, double* a, const int& n0, int* piv,
// int& st);
void dgetrf(const int *n, const int *m, double *a, const int *n0, int *piv,
            int *st);
void dgetri(const int *n, double *a, const int *n0, int const *piv,
            double *work, const int *, int *st);
void dgetrs(const char *trans, const int *n, const int *nrhs, double *a,
            const int *lda, int const *ipiv, double *b, int *ldb, int *info);

void dgemm(const char *TRANSA, const char *TRANSB, const int *M, const int *N,
           const int *K, const double *alpha, const double *A, const int *lda,
           const double *B, const int *ldb, const double *beta, double *C,
           const int *ldc);

void dgemv(const char *TRANS, const int *M, const int *N, const double *alpha,
           const double *A, const int *lda, const double *X, const int *incx,
           const double *beta, double *y, const int *incy);
}
#endif

void apply_shifts(int N, double *hamiltonian, double *overlap, double shift_i,
                  double shift_s) {
  // Note that the first row/column is excluded from the shifts

  for (int i = 1; i < N; i++) {
    hamiltonian[i * N + i] += shift_i;
  }

  for (int i = 1; i < N; i++) {
    for (int j = 1; j < N; j++) {
      hamiltonian[i * N + j] += shift_s * overlap[i * N + j];
    }
  }

  for (int i = 1; i < N; i++) {
    if (overlap[i * N + i] == 0.0)
      overlap[i * N + i] = shift_i * shift_s;
  }
}

void do_inverse(int N, double *matrix) {

  std::vector<int> pivot(N);
  std::vector<double> work(N);

  std::cout << "starting dgetrf" << std::endl;
  int info = 0;
  dgetrf(&N, &N, matrix, &N, pivot.data(), &info);
  if (info != 0) {
    std::cout << "dgetrf error, info = " << info << std::endl;
  }

  info = 0;
  std::cout << "starting dgetri" << std::endl;
  dgetri(&N, matrix, &N, pivot.data(), work.data(), &N, &info);
  if (info != 0) {
    std::cout << "dgetri error, info = " << info << std::endl;
  }
}

// Solve generalized eigenvalue problem with ARPACK

void compute_eigenthings_arpack(int N, double *overlap, double *hamiltonian,
                                double *lowest_ev, double *scaled_evec) {
#if 0
  // do transpose to see if it changes anything
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      std::swap(overlap[N*i + j], overlap[N*j + i]);
      std::swap(hamiltonian[N*i + j], hamiltonian[N*j + i]);
    }
  }
#endif

  // Compute factorization of overlap
  std::vector<int> pivot(N);
  std::vector<double> work(N);

  std::vector<double> overlap_copy(N * N);
  for (int i = 0; i < N * N; i++)
    overlap_copy[i] = overlap[i];

  std::vector<double> tmp(N);

  std::cout << "starting dgetrf" << std::endl;
  int info = 0;
  dgetrf(&N, &N, overlap_copy.data(), &N, pivot.data(), &info);
  if (info != 0) {
    std::cout << "dgetrf error, info = " << info << std::endl;
  }

  arpack::which which_ev = arpack::which::largest_magnitude;

  // Set up for ARPACK
  int ido = 0;
  const int maxit = 200;
  const int nev = 10;
  double tol = 1e-3;
  std::vector<double> resid(N);
  int ncv = 2 * nev + 1;
  int ldv = N;
  std::vector<double> v(ncv * N);
  int lworkl = 3 * ncv * ncv + 6 * ncv;
  std::vector<double> workd(3 * N);
  std::vector<double> workl(lworkl);

  int iparam[11];
  std::vector<int> ipntr(14);
  for (int i = 0; i < 11; i++) {
    iparam[i] = 0;
  }
  iparam[0] = 1;     // ishifts
  iparam[2] = maxit; // maxitr
  iparam[3] = 1;     // nb(blocksize) (should be 1)
  iparam[6] = 2;     // mode

  info = 0;
  for (int i = 0; i < 1000 * maxit; i++) {
    naupd(ido, arpack::bmat::generalized, N,
          // arpack::which::largest_magnitude,
          which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
          ipntr.data(), workd.data(), workl.data(), lworkl, info);

    if (i < 5 || i % 100 == 0)
      std::cout << "it = " << i << " ido = " << ido << std::endl;
    if (info < 0)
      std::cout << "Error, info = " << info << std::endl;
    if (info == 1)
      std::cout << "Max it reached" << std::endl;
    if (ido == 99)
      break;
    if (ido == 1 || ido == -1) {
      //  y = inv(B)*A*x
      //
      info = 0;
      // std::cout << "ipntr1  = " << ipntr[0] << std::endl;
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      // dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, &workd[ipntr[0]-1],
      // &inc, &beta, tmp.data(), &inc);
      dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, &workd[ipntr[0] - 1], &inc,
            &beta, &workd[ipntr[1] - 1], &inc);
      // std::cout << "starting dgetrs" << std::endl;
      int nrhs = 1;
      // dgetrs(&trans, &N, &nrhs, overlap_copy.data(), &N, pivot.data(),
      // tmp.data(), &N, &info);
      dgetrs(&trans, &N, &nrhs, overlap_copy.data(), &N, pivot.data(),
             &workd[ipntr[1] - 1], &N, &info);
      if (info != 0) {
        std::cout << "dgetri error, info = " << info << std::endl;
      }
      // for (int i = 0; i < N; i++) {
      //   workd[ipntr[1]-1 + i] = tmp[i];
      // }
      // std::cout << "done with dgetrs" << std::endl;
    }
    if (ido == 2) {
      // Matrix-vector mult from workd
      // y = B*x
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      int info = 0;
      dgemv(&trans, &N, &N, &alpha, overlap, &N, &workd[ipntr[0] - 1], &inc,
            &beta, &workd[ipntr[1] - 1], &inc);
    }
  }

  // now get the eigenvalues and eigenvectors
  int rvec = true;
  std::vector<int> select(ncv);
  std::vector<double> dr(nev + 1);
  std::vector<double> di(nev + 1);
  std::vector<double> Z(N * (nev + 1));
  int ldz = N;
  double sigmar;
  double sigmai;
  std::vector<double> workev(3 * N);
  arpack::neupd(rvec, arpack::howmny::schur_vectors, select.data(), dr.data(),
                di.data(),
                // Z.data(),
                v.data(), ldz, sigmar, sigmai, workev.data(),
                arpack::bmat::generalized, N,
                // arpack::which::largest_magnitude,
                which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
                ipntr.data(), workd.data(), workl.data(), lworkl, info);

  if (info != 0) {
    std::cout << "neupd error, info = " << info << std::endl;
  }
  std::cout << "nconv (iparam(4)) = " << iparam[4] << std::endl;
  for (int i = 0; i < nev; i++)
    std::cout << "evec i " << i << " " << dr[i] << " imag " << di[i]
              << std::endl;

  *lowest_ev = dr[0];

  for (int i = 0; i < N; i++)
    scaled_evec[i] = v[i] / v[0];

  // Compute residual
  // A*x - lambda*B*x

  std::vector<double> tmp_ax(N);
  std::vector<double> tmp_lbx(N, 0.0);

  char trans('N');
  double alpha = 1.0;
  double beta = 0.0;
  int inc = 1;
  info = 0;
  // dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, scaled_evec, &inc, &beta,
  // tmp_ax.data(), &inc);
  dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, v.data(), &inc, &beta,
        tmp_ax.data(), &inc);

  beta = dr[0];
  // dgemv(&trans, &N, &N, &alpha, overlap, &N, scaled_evec, &inc, &beta,
  // tmp_lbx.data(), &inc);
  dgemv(&trans, &N, &N, &alpha, overlap, &N, v.data(), &inc, &beta,
        tmp_lbx.data(), &inc);
  double resid_norm = 0.0;
  for (int i = 0; i < N; i++) {
    resid[i] = tmp_ax[i] - tmp_lbx[i];
    resid_norm += resid[i] * resid[i];
    if (i < 10)
      std::cout << " resid " << i << " " << resid[i] << " evec = " << v[i]
                << " scaled " << v[i] / v[0] << std::endl;
  }
  std::cout << "Residual :" << resid_norm << std::endl;
}

// Solve generalized eigenvalue problem with ARPACK using the shift-inverse
// method This is the best one to use for QMCPACK

void compute_eigenthings_arpack_shift(int N, double *overlap,
                                      double *hamiltonian, double *lowest_ev,
                                      double *scaled_evec) {
  // do transpose to see if it changes anything (yes, it does help)
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      std::swap(hamiltonian[N * i + j], hamiltonian[N * j + i]);
    }
  }

  // Compute factorization of overlap
  std::vector<int> pivot(N);
  std::vector<double> work(N);

  double zerozero = hamiltonian[0];
  // double sigma = -100.0;
  // double sigma = 0.0;
  double sigma = zerozero * 1.2;
  // double sigma = zerozero - 2.0;
  // double sigma = zerozero;
  std::cout << "sigma = " << sigma << std::endl;

  // A = Hamiltonian
  // B = overlap
  // Compute C = A - sigma*B
  std::vector<double> C_matrix(N * N);
  for (int i = 0; i < N * N; i++)
    C_matrix[i] = hamiltonian[i] - sigma * overlap[i];

  std::cout << "starting dgetrf" << std::endl;
  auto dgetrf_start = std::chrono::system_clock::now();
  int info = 0;
  dgetrf(&N, &N, C_matrix.data(), &N, pivot.data(), &info);
  if (info != 0) {
    std::cout << "dgetrf error, info = " << info << std::endl;
  }
  auto dgetrf_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgetrf_time = dgetrf_end - dgetrf_start;
  std::cout << "  Dgetrf time : " << dgetrf_time.count() << std::endl;

  int ncall_ido1(0);
  int ncall_ido2(0);
  double time_ido1;
  double time_ido2;

  std::vector<double> tmp(N);
  arpack::which which_ev = arpack::which::largest_magnitude;
  // arpack::which which_ev  = arpack::which::largest_real;
  // arpack::which which_ev  = arpack::which::smallest_magnitude;

  // Set up for ARPACK
  int ido = 0;
  const int maxit = 200;
  const int nev = 1;
  double tol = 1e-4;
  std::vector<double> resid(N, 0.1);
  // int ncv = 2*nev + 1;
  int ncv = 61;
  int ldv = N;
  std::vector<double> v(ncv * N);
  int lworkl = 3 * ncv * ncv + 6 * ncv;
  std::vector<double> workd(3 * N);
  std::vector<double> workl(lworkl);

  int iparam[11];
  std::vector<int> ipntr(14);
  for (int i = 0; i < 11; i++) {
    iparam[i] = 0;
  }
  iparam[0] = 1;     // ishifts
  iparam[2] = maxit; // maxitr
  iparam[3] = 1;     // nb(blocksize) (should be 1)
  iparam[6] = 3;     // mode

  auto naupd_start = std::chrono::system_clock::now();

  info = 1;
  for (int it = 0; it < 1000 * maxit; it++) {
    naupd(ido, arpack::bmat::generalized, N,
          // arpack::which::largest_magnitude,
          which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
          ipntr.data(), workd.data(), workl.data(), lworkl, info);

    if (it % 100 == 0)
      std::cout << "it = " << it << " ido = " << ido << std::endl;
    if (info < 0)
      std::cout << "Error, info = " << info << std::endl;
    if (info == 1)
      std::cout << "Max it reached" << std::endl;
    if (ido == 99)
      break;
    if (ido == -1) {
      //  y = inv(C)*B*x
      //
      info = 0;
      // std::cout << "ipntr1  = " << ipntr[0] << std::endl;
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      // dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, &workd[ipntr[0]-1],
      // &inc, &beta, tmp.data(), &inc);
      dgemv(&trans, &N, &N, &alpha, overlap, &N, &workd[ipntr[0] - 1], &inc,
            &beta, &workd[ipntr[1] - 1], &inc);
      // std::cout << "starting dgetrs" << std::endl;
      int nrhs = 1;
      // dgetrs(&trans, &N, &nrhs, overlap_copy.data(), &N, pivot.data(),
      // tmp.data(), &N, &info);
      dgetrs(&trans, &N, &nrhs, C_matrix.data(), &N, pivot.data(),
             &workd[ipntr[1] - 1], &N, &info);
      if (info != 0) {
        std::cout << "dgetrs error, info = " << info << std::endl;
      }
      // for (int i = 0; i < N; i++) {
      //   workd[ipntr[1]-1 + i] = tmp[i];
      // }
      // std::cout << "done with dgetrs" << std::endl;
    }
    if (ido == 1) {
      ncall_ido1++;
      auto ido1_start = std::chrono::system_clock::now();
      //  y = inv(C)*B*x
      // B*x is saved in workd[ipntr[2]-1]
      info = 0;
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      for (int i = 0; i < N; i++)
        workd[ipntr[1] - 1 + i] = workd[ipntr[2] - 1 + i];
      int nrhs = 1;
      dgetrs(&trans, &N, &nrhs, C_matrix.data(), &N, pivot.data(),
             &workd[ipntr[1] - 1], &N, &info);
      if (info != 0) {
        std::cout << "dgetrs error, info = " << info << std::endl;
      }
      auto ido1_end = std::chrono::system_clock::now();
      std::chrono::duration<double> ido1_time = ido1_end - ido1_start;
      if (it < 4)
        std::cout << "time for ido==1 : " << ido1_time.count() << std::endl;
      ;
      time_ido1 += ido1_time.count();
    }
    if (ido == 2) {
      ncall_ido2++;
      auto ido2_start = std::chrono::system_clock::now();
      // Matrix-vector mult from workd
      // y = B*x
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      int info = 0;
      dgemv(&trans, &N, &N, &alpha, overlap, &N, &workd[ipntr[0] - 1], &inc,
            &beta, &workd[ipntr[1] - 1], &inc);
      auto ido2_end = std::chrono::system_clock::now();
      std::chrono::duration<double> ido2_time = ido2_end - ido2_start;
      if (it < 4)
        std::cout << "time for ido==2 : " << ido2_time.count() << std::endl;
      ;
      time_ido2 += ido2_time.count();
    }
  }

  auto naupd_end = std::chrono::system_clock::now();
  std::chrono::duration<double> naupd_time = naupd_end - naupd_start;
  std::cout << "  dnaupd time : " << naupd_time.count() << std::endl;

  std::cout << "ncall ido==1 : " << ncall_ido1 << " ido==2 : " << ncall_ido2
            << std::endl;
  std::cout << "time ido==1 : " << time_ido1 << " ido==2 : " << time_ido2
            << std::endl;

  // now get the eigenvalues and eigenvectors
  int rvec = true;
  // int rvec = false;
  std::vector<int> select(ncv);
  std::vector<double> dr(nev + 1);
  std::vector<double> di(nev + 1);
  std::vector<double> Z(N * (nev + 1));
  int ldz = N;
  double sigmar = sigma;
  double sigmai(0.0);
  std::vector<double> workev(3 * N);
  arpack::neupd(rvec, arpack::howmny::schur_vectors,
                // arpack::howmny::ritz_vectors,
                select.data(), dr.data(), di.data(), Z.data(), ldz, sigmar,
                sigmai, workev.data(), arpack::bmat::generalized, N,
                // arpack::which::largest_magnitude,
                which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
                ipntr.data(), workd.data(), workl.data(), lworkl, info);

  if (info != 0) {
    std::cout << "neupd error, info = " << info << std::endl;
  }
  std::cout << "nconv (iparam(5)) = " << iparam[4] << std::endl;
  for (int i = 0; i < nev; i++) {
    double maybe = sigma + 1.0 / dr[i];
    std::cout << "eval i " << i << " " << dr[i] << " or " << maybe << " imag "
              << di[i] << std::endl;
  }

  //*lowest_ev = sigma + 1.0/dr[0];
  *lowest_ev = dr[0];

  for (int i = 0; i < N; i++)
    scaled_evec[i] = v[i] / v[0];

  // Compute residual
  // A*x - lambda*B*x

  std::vector<double> tmp_ax(N);
  std::vector<double> tmp_lbx(N, 0.0);

  char trans('N');
  double alpha = 1.0;
  double beta = 0.0;
  int inc = 1;
  info = 0;
  // dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, scaled_evec, &inc, &beta,
  // tmp_ax.data(), &inc);
  dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, v.data(), &inc, &beta,
        tmp_ax.data(), &inc);

  // dgemv(&trans, &N, &N, &alpha, overlap, &N, scaled_evec, &inc, &beta,
  // tmp_lbx.data(), &inc);
  dgemv(&trans, &N, &N, &alpha, overlap, &N, v.data(), &inc, &beta,
        tmp_lbx.data(), &inc);
  double resid_norm = 0.0;
  for (int i = 0; i < N; i++) {
    resid[i] = tmp_ax[i] - dr[0] * tmp_lbx[i];
    resid_norm += resid[i] * resid[i];
    if (i < 10)
      std::cout << " resid " << i << " " << resid[i] << " evec = " << v[i]
                << " scaled " << v[i] / v[0] << std::endl;
  }
  std::cout << "Residual :" << resid_norm << std::endl;
}
// Invert the matrix and compute S^-1 H and solve that problem with ARPACK

// The ARPACK documentation methods that if S is positive definite, it might
// be best to perform a Cholesky decomposition of the overlap matrix, transform
// the Hamiltonian ( L H L^T), and solve that problem.

void compute_eigenthings_arpack_regular(int N, double *overlap,
                                        double *hamiltonian, double *lowest_ev,
                                        double *scaled_evec) {
  std::vector<double> prod(N * N);
  auto invert_start = std::chrono::system_clock::now();
  do_inverse(N, overlap);
  auto invert_end = std::chrono::system_clock::now();
  std::chrono::duration<double> invert_time = invert_end - invert_start;
  std::cout << "  Invert matrix time : " << invert_time.count() << std::endl;

  double one(1.0);
  double zero(0.0);

  char transa('N');
  char transb('N');

  auto dgemm_start = std::chrono::system_clock::now();
  dgemm(&transa, &transb, &N, &N, &N, &one, hamiltonian, &N, overlap, &N, &zero,
        prod.data(), &N);
  auto dgemm_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgemm_time = dgemm_end - dgemm_start;
  std::cout << "  Matrix multiply time : " << dgemm_time.count() << std::endl;

  // do transpose (why?)
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      std::swap(prod[N * i + j], prod[N * j + i]);
    }
  }

  arpack::which which_ev = arpack::which::largest_magnitude;
  // arpack::which which_ev  = arpack::which::smallest_algebraic;

  // Set up for ARPACK
  int ido = 0;
  const int maxit = 200;
  const int nev = 10;
  double tol = 1e-3;
  std::vector<double> resid(N);
  int ncv = 2 * nev + 1;
  int ldv = N;
  std::vector<double> v(ncv * N);
  int lworkl = 3 * ncv * ncv + 6 * ncv;
  std::vector<double> workd(3 * N);
  std::vector<double> workl(lworkl);

  int iparam[11];
  std::vector<int> ipntr(14);
  for (int i = 0; i < 11; i++) {
    iparam[i] = 0;
  }
  iparam[0] = 1;     // ishifts
  iparam[2] = maxit; // maxitr
  iparam[3] = 1;     // nb(blocksize) (should be 1)
  iparam[6] = 1;     // mode

  int info = 0;
  for (int i = 0; i < 1000 * maxit; i++) {
    naupd(ido, arpack::bmat::identity, N,
          // arpack::which::largest_magnitude,
          which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
          ipntr.data(), workd.data(), workl.data(), lworkl, info);

    if (i < 5 || i % 100 == 0)
      std::cout << "it = " << i << " ido = " << ido << std::endl;
    if (info != 0)
      std::cout << "Error, info = " << info << std::endl;
    if (info == 1)
      std::cout << "Max it reached" << std::endl;
    if (ido == 99)
      break;
    if (ido == 1 || ido == -1) {
      //  y = (B^-1*A)*x
      //
      info = 0;
      // std::cout << "ipntr1  = " << ipntr[0] << std::endl;
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      // dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, &workd[ipntr[0]-1],
      // &inc, &beta, tmp.data(), &inc);
      dgemv(&trans, &N, &N, &alpha, prod.data(), &N, &workd[ipntr[0] - 1], &inc,
            &beta, &workd[ipntr[1] - 1], &inc);
    }
  }

  // now get the eigenvalues and eigenvectors
  int rvec = true;
  std::vector<int> select(ncv);
  std::vector<double> dr(nev + 1);
  std::vector<double> di(nev + 1);
  std::vector<double> Z(N * (nev + 1));
  int ldz = N;
  double sigmar;
  double sigmai;
  std::vector<double> workev(3 * N);
  arpack::neupd(rvec, arpack::howmny::schur_vectors, select.data(), dr.data(),
                di.data(), Z.data(), ldz, sigmar, sigmai, workev.data(),
                arpack::bmat::identity, N,
                // arpack::which::largest_magnitude,
                which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
                ipntr.data(), workd.data(), workl.data(), lworkl, info);

  if (info < 0) {
    std::cout << "neupd error, info = " << info << std::endl;
  }
  std::cout << "nconv (iparam(4)) = " << iparam[4] << std::endl;
  for (int i = 0; i < nev; i++)
    std::cout << "evec i " << i << " " << dr[i] << " imag " << di[i]
              << std::endl;

  *lowest_ev = dr[0];

  for (int i = 0; i < N; i++)
    scaled_evec[i] = v[i] / v[0];

  // Compute residual
  // A*x - lambda*x

  std::vector<double> tmp_ax(N);

  char trans('N');
  double alpha = 1.0;
  double beta = 0.0;
  int inc = 1;
  info = 0;
  // dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, scaled_evec, &inc, &beta,
  // tmp_ax.data(), &inc);
  dgemv(&trans, &N, &N, &alpha, prod.data(), &N, v.data(), &inc, &beta,
        tmp_ax.data(), &inc);

  double resid_norm = 0.0;
  for (int i = 0; i < N; i++) {
    resid[i] = tmp_ax[i] - dr[0] * v[i];
    resid_norm += resid[i] * resid[i];
    if (i < 10)
      std::cout << " resid " << i << " " << resid[i] << " evec = " << v[i]
                << " scaled " << v[i] / v[0] << std::endl;
  }
  std::cout << "Residual :" << resid_norm << std::endl;
}

// Invert the overlap matrix and compute S^-1 H and solve that problem with
// ARPACK using the shift-inverse method
void compute_eigenthings_arpack_regular_shift(int N, double *overlap,
                                              double *hamiltonian,
                                              double *lowest_ev,
                                              double *scaled_evec) {
  std::vector<double> overlap_copy(N * N);
  for (int i = 0; i < N * N; i++)
    overlap_copy[i] = overlap[i];

  std::vector<double> prod(N * N);
  auto invert_start = std::chrono::system_clock::now();
  do_inverse(N, overlap);
  auto invert_end = std::chrono::system_clock::now();
  std::chrono::duration<double> invert_time = invert_end - invert_start;
  std::cout << "  Invert matrix time : " << invert_time.count() << std::endl;

  double one(1.0);
  double zero(0.0);

  char transa('N');
  char transb('N');

  auto dgemm_start = std::chrono::system_clock::now();
  dgemm(&transa, &transb, &N, &N, &N, &one, hamiltonian, &N, overlap, &N, &zero,
        prod.data(), &N);
  // dgemm(&transa, &transb, &N, &N, &N, &one, overlap, &N, hamiltonian, &N,
  // &zero, prod.data(), &N);
  auto dgemm_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgemm_time = dgemm_end - dgemm_start;
  std::cout << "  Matrix multiply time : " << dgemm_time.count() << std::endl;

#if 0
  // do transpose (why?)
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      std::swap(prod[N*i + j], prod[N*j + i]);
    }
  }
#endif

  // Compute factorization of C_matrix
  std::vector<int> pivot(N);
  std::vector<double> work(N);

  double zerozero = hamiltonian[0];
  // double sigma = -100.0;
  // double sigma = 0.0;
  double sigma = zerozero * 1.2;
  // double sigma = zerozero - 2.0;
  // double sigma = zerozero;
  std::cout << "sigma = " << sigma << std::endl;

  // A = Hamiltonian
  // B = overlap
  // Compute C = B^-1 * A - sigma*I
  std::vector<double> C_matrix(N * N);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (i == j)
        C_matrix[i * N + j] = prod[i * N + j] - sigma;
      else
        C_matrix[i * N + j] = prod[i * N + j];

  std::cout << "starting dgetrf" << std::endl;
  int info = 0;
  dgetrf(&N, &N, C_matrix.data(), &N, pivot.data(), &info);
  if (info != 0) {
    std::cout << "dgetrf error, info = " << info << std::endl;
  }

  arpack::which which_ev = arpack::which::largest_magnitude;
  // arpack::which which_ev  = arpack::which::smallest_algebraic;

  // Set up for ARPACK
  int ido = 0;
  const int maxit = 200;
  const int nev = 10;
  double tol = 1e-3;
  std::vector<double> resid(N);
  int ncv = 2 * nev + 1;
  int ldv = N;
  std::vector<double> v(ncv * N);
  int lworkl = 3 * ncv * ncv + 6 * ncv;
  std::vector<double> workd(3 * N);
  std::vector<double> workl(lworkl);

  int iparam[11];
  std::vector<int> ipntr(14);
  for (int i = 0; i < 11; i++) {
    iparam[i] = 0;
  }
  iparam[0] = 1;     // ishifts
  iparam[2] = maxit; // maxitr
  iparam[3] = 1;     // nb(blocksize) (should be 1)
  iparam[6] = 3;     // mode

  info = 0;
  for (int i = 0; i < 1000 * maxit; i++) {
    naupd(ido, arpack::bmat::identity, N,
          // arpack::which::largest_magnitude,
          which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
          ipntr.data(), workd.data(), workl.data(), lworkl, info);

    std::cout << "it = " << i << " ido = " << ido << std::endl;
    if (info < 0)
      std::cout << "Error, info = " << info << std::endl;
    if (info == 1)
      std::cout << "Max it reached" << std::endl;
    if (ido == 99)
      break;
    if (ido == 1 || ido == -1) {
      //  y = inv(B^-1*A - sigma*I)*x
      //
      info = 0;
      char trans('N');
      double alpha = 1.0;
      double beta = 0.0;
      int inc = 1;
      for (int i = 0; i < N; i++)
        workd[ipntr[1] - 1 + i] = workd[ipntr[2] - 1 + i];
      int nrhs = 1;
      dgetrs(&trans, &N, &nrhs, C_matrix.data(), &N, pivot.data(),
             &workd[ipntr[1] - 1], &N, &info);
      if (info != 0) {
        std::cout << "dgetrs error, info = " << info << std::endl;
      }
    }
  }

  // now get the eigenvalues and eigenvectors
  int rvec = true;
  std::vector<int> select(ncv);
  std::vector<double> dr(nev + 1);
  std::vector<double> di(nev + 1);
  std::vector<double> Z(N * (nev + 1));
  int ldz = N;
  double sigmar = sigma;
  double sigmai(0.0);
  std::vector<double> workev(3 * N);
  arpack::neupd(rvec, arpack::howmny::schur_vectors, select.data(), dr.data(),
                di.data(), Z.data(), ldz, sigmar, sigmai, workev.data(),
                arpack::bmat::identity, N,
                // arpack::which::largest_magnitude,
                which_ev, nev, tol, resid.data(), ncv, v.data(), ldv, iparam,
                ipntr.data(), workd.data(), workl.data(), lworkl, info);

  if (info < 0) {
    std::cout << "neupd error, info = " << info << std::endl;
  }
  std::cout << "nconv (iparam(4)) = " << iparam[4] << std::endl;
  for (int i = 0; i < nev; i++)
    std::cout << "evec i " << i << " " << dr[i] << " imag " << di[i]
              << std::endl;

  *lowest_ev = dr[0];

  for (int i = 0; i < N; i++)
    scaled_evec[i] = v[i] / v[0];

  // Compute inv(B)*y
  std::vector<double> evec2(N);
  char trans('N');
  char transT('T');
  double alpha = 1.0;
  double beta = 0.0;
  int inc = 1;
  info = 0;
  dgemv(&trans, &N, &N, &alpha, overlap, &N, v.data(), &inc, &beta,
        evec2.data(), &inc);
  // dgemv(&transT, &N, &N, &alpha, overlap, &N, v.data(), &inc, &beta,
  // evec2.data(), &inc);

  // Compute residual
  // A*x - lambda*B*x

  std::vector<double> tmp_ax(N);
  std::vector<double> tmp_bx(N);

#if 0
    char trans('N');
    double alpha = 1.0;
    double beta = 0.0;
    int inc = 1;
    info = 0;
#endif
  // dgemv(&trans, &N, &N, &alpha, prod.data(), &N, v.data(), &inc, &beta,
  // tmp_ax.data(), &inc);

  dgemv(&trans, &N, &N, &alpha, hamiltonian, &N, evec2.data(), &inc, &beta,
        tmp_ax.data(), &inc);
  dgemv(&trans, &N, &N, &alpha, overlap_copy.data(), &N, evec2.data(), &inc,
        &beta, tmp_bx.data(), &inc);

  double resid_norm = 0.0;
  for (int i = 0; i < N; i++) {
    // resid[i] = tmp_ax[i] - dr[0]*v[i];
    resid[i] = tmp_ax[i] - dr[0] * tmp_bx[i];
    resid_norm += resid[i] * resid[i];
    if (i < 10)
      // std::cout << " resid " << i << " " << resid[i] << " evec = " << v[i] <<
      // " scaled " << v[i]/v[0] << std::endl;
      std::cout << " resid " << i << " " << resid[i] << " evec orig " << v[i]
                << " evec = " << evec2[i] << " scaled " << evec2[i] / evec2[0]
                << std::endl;
  }
  std::cout << "Residual :" << resid_norm << std::endl;
}

// There are many variants of the linear method that adjust the Hamiltonian
// matrix to improve stability of the optimization. This follows the
// "one_shift_only" code in QMCFixedSampleLinearOptimizeBatched.cpp

int main(int argc, char **argv) {

  // std::string fname = "linear_matrices.h5";
  // std::string fname = "propane5k_linear_matrices.h5";
  std::string fname = "propane100_linear_matrices.h5";
  // std::string fname = "prop_three_body.h5";
  // std::string fname = "methane_linear_matrices.h5";

  if (argc > 1) {
    fname = argv[1];
  }

  std::cout << "Using file: " << fname << std::endl;

  hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cout << "Unable to open: " << fname << std::endl;
    return 1;
  }

  std::chrono::system_clock::time_point read_start =
      std::chrono::system_clock::now();

  hid_t h1 = H5Dopen(file_id, "overlap", H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(h1);
  int rank = H5Sget_simple_extent_ndims(dataspace);
  std::cout << " rank = " << rank << std::endl;

  std::vector<hsize_t> sizes(rank);
  H5Sget_simple_extent_dims(dataspace, sizes.data(), NULL);
  std::cout << " size = " << sizes[0] << std::endl;

  int N = sizes[0];

  hsize_t matrix_size = sizes[0] * sizes[1];
  std::vector<double> overlap_data(matrix_size);
  H5Dread(h1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          overlap_data.data());
  std::cout << "overlap = " << overlap_data[0] << " " << overlap_data[11]
            << std::endl;

  std::vector<double> hamiltonian_data(matrix_size);
  hid_t h2 = H5Dopen(file_id, "Hamiltonian", H5P_DEFAULT);
  H5Dread(h2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          hamiltonian_data.data());

  std::cout << "ham = " << hamiltonian_data[0] << " " << hamiltonian_data[11]
            << std::endl;

  double qmcpack_lowest_ev;
  hid_t h3 = H5Dopen(file_id, "lowest_eigenvalue", H5P_DEFAULT);
  H5Dread(h3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          &qmcpack_lowest_ev);

  std::vector<double> qmcpack_scaled_evec(N);
  hid_t h4 = H5Dopen(file_id, "scaled_eigenvector", H5P_DEFAULT);
  H5Dread(h4, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          qmcpack_scaled_evec.data());

  double shift_i;
  hid_t h5 = H5Dopen(file_id, "bestShift_i", H5P_DEFAULT);
  H5Dread(h5, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &shift_i);

  double shift_s;
  hid_t h6 = H5Dopen(file_id, "bestShift_s", H5P_DEFAULT);
  H5Dread(h6, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &shift_s);

  H5Fclose(file_id);

  std::chrono::system_clock::time_point read_end =
      std::chrono::system_clock::now();
  std::chrono::duration<double> read_time = read_end - read_start;
  std::cout << "File read time: " << read_time.count() << std::endl;

  apply_shifts(N, hamiltonian_data.data(), overlap_data.data(), shift_i,
               shift_s);

  double lowest_ev;
  std::vector<double> scaled_evec(N);

  auto eig_start = std::chrono::system_clock::now();

  // compute_eigenthings_arpack(N, overlap_data.data(), hamiltonian_data.data(),
  // &lowest_ev, scaled_evec.data());
  compute_eigenthings_arpack_shift(N, overlap_data.data(),
                                   hamiltonian_data.data(), &lowest_ev,
                                   scaled_evec.data());
  // compute_eigenthings_arpack_regular(N, overlap_data.data(),
  // hamiltonian_data.data(), &lowest_ev, scaled_evec.data());
  // compute_eigenthings_arpack_regular_shift(N, overlap_data.data(),
  // hamiltonian_data.data(), &lowest_ev, scaled_evec.data());
  auto eig_end = std::chrono::system_clock::now();
  std::chrono::duration<double> eig_time = eig_end - eig_start;
  std::cout << "Total generalized eigenvalue time: " << eig_time.count()
            << std::endl;

  std::cout << std::setprecision(10) << "lowest ev: qmcpack, this, diff "
            << qmcpack_lowest_ev << " " << lowest_ev << " "
            << (qmcpack_lowest_ev - lowest_ev) << std::endl;

  double norm_sum = 0.0;
  for (int i = 0; i < N; i++) {
    double diff = scaled_evec[i] - qmcpack_scaled_evec[i];
    norm_sum += diff * diff;
  }

  double norm = std::sqrt(norm_sum);
  std::cout << " norm diff scaled evec: " << norm << std::endl;

  return 0;
}
