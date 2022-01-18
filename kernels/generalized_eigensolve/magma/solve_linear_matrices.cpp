

#include <string>
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>

//#define USE_MKL
#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_MAGMA
#include <magma_v2.h>
#include <magma_lapack.h>
#endif


#ifndef USE_MKL
extern "C" {
#define dgetrf dgetrf_
#define dgetri dgetri_
#define dgeev dgeev_
#define dggev dggev_
#define dgemm dgemm_
//void dgetrf(const int& n, const int& m, double* a, const int& n0, int* piv, int& st);
void dgetrf(const int* n, const int* m, double* a, const int* n0, int* piv, int* st);
void dgetri(const int* n, double* a, const int* n0, int const* piv, double* work, const int*, int* st);

  void dgeev(const char* JOBVL,
             const char* JOBVR,
             const int* N,
             double* A,
             const int* LDA,
             double* ALPHAR,
             double* ALPHAI,
             double* VL,
             const int* LDVL,
             double* VR,
             const int* LDVR,
             double* WORK,
             const int* LWORK,
             int* INFO);

  void dggev(char* JOBVL,
             char* JOBVR,
             int* N,
             double* A,
             int* LDA,
             double* B,
             int* LDB,
             double* ALPHAR,
             double* ALPHAI,
             double *BETA,
             double* VL,
             int* LDVL,
             double* VR,
             int* LDVR,
             double* WORK,
             int* LWORK,
             int* INFO);

  void dgemm(const char* TRANSA,
             const char* TRANSB,
             const int* M,
             const int* N,
             const int* K,
             const double* alpha,
             const double* A,
             const int* lda,
             const double* B,
             const int* ldb,
             const double* beta,
             double* C,
             const int* ldc);


}
#endif

void apply_shifts(int N, double *hamiltonian, double *overlap, double shift_i, double shift_s)
{
  // Note that the first row/column is excluded from the shifts
  
  for (int i = 1; i < N; i++) {
    hamiltonian[i*N + i] += shift_i;
  }

  for (int i = 1; i < N; i++) {
    for (int j = 1; j < N; j++) {
      hamiltonian[i*N + j]  += shift_s * overlap[i*N + j];
    }
  }
}

void do_inverse_magma(int N, double *matrix, magma_queue_t queue)
{

  double* d_A;
  magma_int_t ret = magma_dmalloc(&d_A, N*N);
  if (ret != MAGMA_SUCCESS) {
    std::cout << "magma dmalloc for d_A failed\n";
  }
  magma_dsetmatrix(N, N, matrix, N, d_A, N, queue);

  auto dgetrf_start = std::chrono::system_clock::now();
  std::vector<int> pivot(N);
  int info;
  int status = magma_dgetrf_gpu(
      N,
      N,
      d_A,
      N,
      pivot.data(),
      &info
      );
  auto dgetrf_end = std::chrono::system_clock::now();

  std::cout << " status from dgetrf = " << status << std::endl;
  std::cout << " info from degetrf = " << info << std::endl;
  std::chrono::duration<double> dgetrf_time = dgetrf_end - dgetrf_start;
  std::cout << "    dgetrf time : " << dgetrf_time.count() << std::endl;

  int nb = magma_get_dgetri_nb(N);
  int lwork = nb*N;
  std::cout << " lwork = " << lwork << std::endl;

  double* d_work;
  ret = magma_dmalloc(&d_work, lwork);
  if (ret != MAGMA_SUCCESS) {
    std::cout << "magma dmalloc for dwork failed\n";
  }

  auto dgetri_start = std::chrono::system_clock::now();

  status = magma_dgetri_gpu(
      N,
      d_A,
      N,
      pivot.data(),
      d_work,
      lwork,
      &info);
  auto dgetri_end = std::chrono::system_clock::now();


  std::cout << " status from dgetri = " << status << std::endl;
  std::cout << " info from degetri = " << info << std::endl;

  std::chrono::duration<double> dgetri_time = dgetri_end - dgetri_start;
  std::cout << "    dgetri time : " << dgetri_time.count() << std::endl;

  magma_dgetmatrix(N, N, d_A, N, matrix, N, queue);



}

void do_inverse(int N, double *matrix)
{

  std::vector<int> pivot(N);
  std::vector<double> work(N);

  std::cout << "starting dgetrf" << std::endl;
  int info;
  dgetrf(&N, &N, matrix, &N, pivot.data(), &info);
  if (info != 0) {
    std::cout << "dgetrf error, info = " << info << std::endl;
  }

  std::cout << "starting dgetri" << std::endl;
  dgetri(&N, matrix, &N, pivot.data(), work.data(), &N, &info);
  if (info != 0) {
    std::cout << "dgetri error, info = " << info << std::endl;
  }

}

// Solve generalized eigenvalue problem using LAPACK solver
void compute_eigenthings_generalized(int N, double* overlap, double* hamiltonian, double *lowest_ev, double* scaled_eval)
{
  std::vector<double> alphar(N);
  std::vector<double> alphai(N);
  std::vector<double> beta(N);
  std::vector<double> vl(N*N);
  std::vector<double> vr(N*N);

  // first get the lwork size
  int lwork = -1;
  std::vector<double> work(1);
  int info;
  char jl('V');
  char jr('V');

  dggev(&jl,&jr, &N, overlap, &N, hamiltonian, &N,
        alphar.data(), alphai.data(), beta.data(), vl.data(), &N, vr.data(), &N, work.data(), &lwork, &info);

  lwork = work[0];
  std::cout << "optimal lwork = " << lwork << std::endl;
  work.resize(lwork);

  //dggev(&jl,&jr, &N, overlap, &N, hamiltonian, &N,
  //      alphar.data(), alphai.data(), beta.data(), vl.data(), &N, vr.data(), &N,
  //      work.data(), &lwork, &info);
  dggev(&jl,&jr, &N, hamiltonian, &N, overlap, &N,
        alphar.data(), alphai.data(), beta.data(), vl.data(), &N, vr.data(), &N,
        work.data(), &lwork, &info);
  if (info != 0) {
    std::cout << "dggev error, info = " << info << std::endl;
  }


  std::cout << "Eigenvalues" << std::endl;
  int idx = 0;
  double min_eig = 1000;
  for (int i = 0; i < N; i++) {
    double eig = alphar[i]/beta[i];
    //std::cout << i << " " << eig << std::endl;
    if (eig < min_eig) {
      idx = i;
      min_eig = eig;
    }
  }

  std::cout << "Index of min eigenvalue = " << idx << std::endl;

  *lowest_ev = alphar[idx]/beta[idx];


#if 0
  std::cout << "Eigenvector (right)" << std::endl;
  for (int i = 0; i < N; i++) {
    double eval = vr[idx*N + i];
    double scaled_eval = eval/vr[0];
    std::cout << i << " " << eval << " " << scaled_eval << std::endl;
  }
#endif


  std::cout << "Eigenvector (left)" << std::endl;
  for (int i = 0; i < N; i++) {
    double eval = vl[idx*N + i];
    scaled_eval[i] = eval/vl[idx*N];
    //std::cout << i << " " << eval << " " << scaled_eval[i] << std::endl;
  }


}

// Solve generalized eigenvalue problem by inverse and multiply
//  (that is, the way numerical analysists tell you not to solve it)
void compute_eigenthings_other(int N, double* overlap, double* hamiltonian, double *lowest_ev, double* scaled_eval, magma_queue_t queue)
{

  std::vector<double> prod(N*N);
  auto invert_start = std::chrono::system_clock::now();
  //do_inverse(N, overlap);
  do_inverse_magma(N, overlap, queue);
  auto invert_end = std::chrono::system_clock::now();
  std::chrono::duration<double> invert_time = invert_end - invert_start;
  std::cout << "  Invert matrix time : " << invert_time.count() << std::endl;



  double one(1.0);
  double zero(0.0);

  char transa('N');
  char transb('N');

  auto dgemm_start = std::chrono::system_clock::now();
  dgemm(&transa, &transb, &N, &N, &N, &one, hamiltonian, &N, overlap, &N, &zero, prod.data(), &N);
  auto dgemm_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgemm_time = dgemm_end - dgemm_start;
  std::cout << "  Matrix multiply time : " << dgemm_time.count() << std::endl;


  // do transpose (why?)
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      std::swap(prod[N*i + j], prod[N*j + i]);
    }
  }

  double zerozero = prod[0];


  std::vector<double> alphar(N);
  std::vector<double> alphai(N);
  std::vector<double> vl(N*N);
  std::vector<double> vr(N*N);

  char jl('N');
  char jr('V');

  int info;

  int lwork = -1;
  std::vector<double> work(1);
#ifdef USE_MAGMA
  magma_dgeev(MagmaNoVec, MagmaVec, N, prod.data(), N,
        alphar.data(), alphai.data(), vl.data(), N, vr.data(), N,
        work.data(), lwork, &info);
#else
  dgeev(&jl,&jr, &N, prod.data(), &N,
        alphar.data(), alphai.data(), vl.data(), &N, vr.data(), &N,
        work.data(), &lwork, &info);
#endif

  lwork = work[0];
  std::cout << "optimal lwork = " << lwork << std::endl;
  work.resize(lwork);

  auto dgeev_start = std::chrono::system_clock::now();
#ifdef USE_MAGMA
  magma_dgeev(MagmaNoVec, MagmaVec, N, prod.data(), N,
        alphar.data(), alphai.data(), vl.data(), N, vr.data(), N,
        work.data(), lwork, &info);
#else
  dgeev(&jl,&jr, &N, prod.data(), &N,
        alphar.data(), alphai.data(), vl.data(), &N, vr.data(), &N,
        work.data(), &lwork, &info);
#endif

  if (info != 0) {
    std::cout << "dgeev error, info = " << info << std::endl;
  }
  auto dgeev_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgeev_time = dgeev_end - dgeev_start;
  std::cout << "  Eigenvalue solver time : " << dgeev_time.count() << std::endl;


#if 0
  std::cout << "Eigenvalues" << std::endl;
  for (int i = 0; i < N; i++) {
    double eig = alphar[i];
    std::cout << std::setprecision(10) << i << " " << eig << std::endl;
  }
#endif

#if 0
  std::vector<double>::iterator min_elem = std::min_element(alphar.begin(), alphar.end());
  int idx = std::distance(alphar.begin(), min_elem);
  *lowest_ev = alphar[idx];
#endif

  std::vector<double> adjusted_ev(N);
  std::transform(alphar.begin(), alphar.end(), adjusted_ev.begin(),
      [&] (double eig) -> double {return (eig - zerozero + 2.0)*(eig - zerozero + 2.0);});

  std::vector<double>::iterator min_elem = std::min_element(adjusted_ev.begin(), adjusted_ev.end());
  int idx = std::distance(adjusted_ev.begin(), min_elem);
  *lowest_ev = alphar[idx];


  std::cout << "Index of min eigenvalue = " << idx << std::endl;
  std::cout << "Eigenvectors" << std::endl;
  for (int i = 0; i < N; i++) {
    double eval = vr[idx*N + i];
    scaled_eval[i] = eval/vr[idx*N];
    //std::cout << i << " " << eval << " " << scaled_eval[i] << std::endl;
  }

}


// There are many variants of the linear method that adjust the Hamiltonian matrix 
// to improve stability of the optimization.
// This follows the "one_shift_only" code in QMCFixedSampleLinearOptimizeBatched.cpp


int main(int argc, char **argv)
{

  //std::string fname = "linear_matrices.h5";
  //std::string fname = "propane5k_linear_matrices.h5";
  std::string fname = "propane100_linear_matrices.h5";
  //std::string fname = "prop_three_body.h5";
  //std::string fname = "methane_linear_matrices.h5";

  if (argc > 1) {
    fname = argv[1];
  }

  std::cout << "Using file: " << fname << std::endl;

  hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cout << "Unable to open: " << fname << std::endl;
    return 1;
  }

  magma_init();
  magma_queue_t queue;
  magma_int_t dev = 0;
  magma_queue_create(dev, &queue);


  std::chrono::system_clock::time_point read_start = std::chrono::system_clock::now();

  hid_t h1 = H5Dopen(file_id,"overlap",H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(h1);
  int rank = H5Sget_simple_extent_ndims(dataspace);
  std::cout << " rank = " << rank << std::endl;

  std::vector<hsize_t> sizes(rank);
  H5Sget_simple_extent_dims(dataspace, sizes.data(), NULL);
  std::cout << " size = " << sizes[0] << std::endl;

  int N = sizes[0];

  hsize_t matrix_size= sizes[0] * sizes[1];
  std::vector<double> overlap_data(matrix_size);
  H5Dread(h1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, overlap_data.data());
  std::cout <<"overlap = " << overlap_data[0] <<  " " << overlap_data[11] << std::endl;


  std::vector<double> hamiltonian_data(matrix_size);
  hid_t h2 = H5Dopen(file_id,"Hamiltonian",H5P_DEFAULT);
  H5Dread(h2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hamiltonian_data.data());

  std::cout <<"ham = " << hamiltonian_data[0] <<  " " << hamiltonian_data[11] << std::endl;

  double qmcpack_lowest_ev;
  hid_t h3 = H5Dopen(file_id,"lowest_eigenvalue",H5P_DEFAULT);
  H5Dread(h3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &qmcpack_lowest_ev);

  std::vector<double> qmcpack_scaled_evec(N);
  hid_t h4 = H5Dopen(file_id,"scaled_eigenvector",H5P_DEFAULT);
  H5Dread(h4, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, qmcpack_scaled_evec.data());


  double shift_i;
  hid_t h5 = H5Dopen(file_id,"bestShift_i",H5P_DEFAULT);
  H5Dread(h5, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &shift_i);

  double shift_s;
  hid_t h6 = H5Dopen(file_id,"bestShift_s",H5P_DEFAULT);
  H5Dread(h6, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &shift_s);


  H5Fclose(file_id);

  std::chrono::system_clock::time_point read_end = std::chrono::system_clock::now();
  std::chrono::duration<double> read_time = read_end - read_start;
  std::cout << "File read time: " << read_time.count() << std::endl;

  apply_shifts(N, hamiltonian_data.data(), overlap_data.data(), shift_i, shift_s);


  double lowest_ev;
  std::vector<double> scaled_evec(N);

  auto eig_start = std::chrono::system_clock::now();

  //do_inverse(overlap_data.data(), sizes[0]);
  //compute_eigenthings_generalized(N, overlap_data.data(), hamiltonian_data.data(), &lowest_ev, scaled_evec.data());
  //
  compute_eigenthings_other(N, overlap_data.data(), hamiltonian_data.data(), &lowest_ev, scaled_evec.data(), queue);
  auto eig_end = std::chrono::system_clock::now();
  std::chrono::duration<double> eig_time = eig_end - eig_start;
  std::cout << "Total generalized eigenvalue time: " << eig_time.count() << std::endl;


  std::cout << std::setprecision(10) << "lowest ev: qmcpack, this, diff " << qmcpack_lowest_ev << " " << lowest_ev << " " << (qmcpack_lowest_ev - lowest_ev) << std::endl;

  double norm_sum = 0.0;
  for (int i = 0; i < N; i++) {
    double diff = scaled_evec[i] - qmcpack_scaled_evec[i];
    norm_sum += diff*diff;
  }

  double norm = std::sqrt(norm_sum);
  std::cout << " norm diff scaled evec: " << norm << std::endl;

  magma_queue_destroy(queue);
  magma_finalize();

  return 0;
}
