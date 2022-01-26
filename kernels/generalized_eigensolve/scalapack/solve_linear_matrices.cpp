
// Scalapack version
//  Needs MKL which has pdgeevx implementation

#include <string>
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include <mpi.h>

//#define USE_MKL
#ifdef USE_MKL
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#endif

const int dlen = 9;


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

void do_inverse(int N, std::vector<int>& desc, double *matrix, int myrank)
{

  std::vector<int> ipiv(N);

  if (myrank == 0) std::cout << "starting dgetrf" << std::endl;
  int info;
  int ia = 1;
  int ja = 1;
  pdgetrf_(&N, &N, matrix, &ia, &ja, desc.data(), ipiv.data(), &info);

  //dgetrf(&N, &N, matrix, &N, pivot.data(), &info);
  if (info != 0) {
    std::cout << "pdgetrf error, info = " << info << std::endl;
  }


  // Query for workspace sizes

  int lwork = -1;
  int liwork = -1;
  std::vector<double> work(1);
  std::vector<int> iwork(1);
  pdgetri_(&N, matrix, &ia, &ja, desc.data(), ipiv.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
  if (info != 0) {
    std::cout << "info from pdgetri size query: " << info << std::endl;
  }

  lwork = work[0];
  liwork = iwork[0];
  if (myrank == 0) std::cout <<" lwork = " << lwork << " liwork = " << liwork << std::endl;

  work.resize(lwork);
  iwork.resize(liwork);


  if (myrank == 0) std::cout << "starting dgetri" << std::endl;
  // Now do the real call

  pdgetri_(&N, matrix, &ia, &ja, desc.data(), ipiv.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

  if (info != 0) {
    std::cout << "pdgetri error, info = " << info << std::endl;
  }

}

#if 0
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
#endif

// Solve generalized eigenvalue problem by inverse and multiply
//  (that is, the way numerical analysists tell you not to solve it)
void compute_eigenthings_other(int N, double* overlap, double* hamiltonian, double *lowest_ev, double* scaled_eval)
{

  int myrank;
  int Nproc;
  blacs_pinfo(&myrank, &Nproc);
  if (myrank == 0) {
    std::cout << "Available procs = " << Nproc << std::endl;
  }

  //const int Nprow = 2;
  //const int Npcol = 2;
  int ictxt = -1;
  //sl_init_(&ictxt, &Nprow, &Npcol);

  int ictxt_blacs;
  const int what = 0;
  blacs_get_(&ictxt, &what, &ictxt_blacs);


  int np = std::sqrt(Nproc);
  if (myrank == 0) {
    std::cout << " np = " << np << std::endl;
  }
  const int Nproc_row = np;
  const int Nproc_col = np;
  char order = 'R';
  blacs_gridinit_(&ictxt_blacs, &order, &Nproc_row, &Nproc_col);

  int myproc_row;
  int myproc_col;
  int tmp_nproc_row;
  int tmp_nproc_col;
  blacs_gridinfo_(&ictxt_blacs, &tmp_nproc_row, &tmp_nproc_col, &myproc_row, &myproc_col);

  // Blocking parameter
  //int bl = N/np;
  int bl = 64;
  int irow = 0;
  int icol = 0;
  //int Nl = N/Nproc_row; // N local to rank
  int firstrank = 0;
  int Nl_row = numroc_(&N, &bl, &myproc_row, &firstrank, &Nproc_row);
  int Nl_col = numroc_(&N, &bl, &myproc_col, &firstrank, &Nproc_col);

  std::cout <<"rank = " << myrank << " Nlrow = " << Nl_row << " Nlcol = " << Nl_col << std::endl;

  // descriptor for distributed matrix
  int info;
  std::vector<int> desc_A(dlen);
  descinit_(desc_A.data(), &N, &N, &bl, &bl, &irow, &icol, &ictxt_blacs, &Nl_row, &info );
  if (info != 0) {
    std::cout << "info from descinit: " << info << std::endl;
  }

  // descriptor for local matrix
  std::vector<int> desc_B(dlen);
  int global_bl = N*N;
  descinit_(desc_B.data(), &N, &N, &N, &global_bl, &irow, &icol, &ictxt_blacs, &N, &info );
  if (info != 0) {
    std::cout << "info from descinit B: " << info << std::endl;
  }

  std::vector<double> local_overlap(Nl_row*Nl_col);
  std::vector<double> local_H(Nl_row*Nl_col);
  std::vector<double> local_prod(Nl_row*Nl_col);
  std::vector<double> local_trans(Nl_row*Nl_col);

  int ia = 1;
  int ja = 1;

  int ib = 1;
  int jb = 1;

  pdgemr2d_(&N, &N, overlap, &ia, &ja, desc_B.data(), local_overlap.data(), &ib, &jb, desc_A.data(), &ictxt_blacs);



  auto invert_start = std::chrono::system_clock::now();
  do_inverse(N, desc_A, local_overlap.data(), myrank);
  auto invert_end = std::chrono::system_clock::now();
  std::chrono::duration<double> invert_time = invert_end - invert_start;
  if (myrank == 0) std::cout << "  Invert matrix time : " << invert_time.count() << std::endl;


  pdgemr2d_(&N, &N, hamiltonian, &ia, &ja, desc_B.data(), local_H.data(), &ib, &jb, desc_A.data(), &ictxt_blacs);

  double one(1.0);
  double zero(0.0);

  char transa('N');
  char transb('N');


  auto dgemm_start = std::chrono::system_clock::now();
  //dgemm(&transa, &transb, &N, &N, &N, &one, hamiltonian, &N, overlap, &N, &zero, prod.data(), &N);
  pdgemm(&transa, &transb, &N, &N, &N, &one, local_H.data(), &ia, &ja, desc_A.data(), local_overlap.data(), &ia, &ja, desc_A.data(), &zero, local_prod.data(), &ia, &ja, desc_A.data());
  auto dgemm_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgemm_time = dgemm_end - dgemm_start;
  if (myrank == 0) std::cout << "  Matrix multiply time : " << dgemm_time.count() << std::endl;


  // do transpose (why?)
#if 0
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      std::swap(prod[N*i + j], prod[N*j + i]);
    }
  }
#endif
  pdtran(&N, &N, &one, local_prod.data(), &ia, &ja, desc_A.data(), &zero, local_trans.data(), &ia, &ja, desc_A.data());

  //double zerozero = prod[0];
  double zerozero = local_trans[0];
  MPI_Bcast(&zerozero, 1, MPI_INT, 0, MPI_COMM_WORLD);


  std::vector<double> alphar(N);
  std::vector<double> alphai(N);
  std::vector<double> vl(1);
  std::vector<double> vr(1);
  if (myrank == 0) {
    vr.resize(N*N);
  }

  std::vector<double> local_vr(Nl_row*Nl_col);

  char jl('N');
  char jr('V');

  int lwork = -1;
  std::vector<double> work(1);

  //dgeev(&jl,&jr, &N, prod.data(), &N,
  //      alphar.data(), alphai.data(), vl.data(), &N, vr.data(), &N,
  //      work.data(), &lwork, &info);

  char balanc('N');
  char sense('N');
  int ilo=1;
  int ihi=N;
  double scale;
  double abnrm;
  double rconde;
  double rcondv;

  pdgeevx(&balanc, &jl, &jr, &sense, &N, local_trans.data(), desc_A.data(),
          alphar.data(), alphai.data(), vl.data(), desc_A.data(), local_vr.data(), desc_A.data(),
          &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv, work.data(), &lwork, &info);
  lwork = work[0];
  std::cout << "optimal lwork = " << lwork << std::endl;
  work.resize(lwork);

  auto dgeev_start = std::chrono::system_clock::now();
  //dgeev(&jl,&jr, &N, prod.data(), &N,
  //      alphar.data(), alphai.data(), vl.data(), &N, vr.data(), &N,
  //      work.data(), &lwork, &info);
  pdgeevx(&balanc, &jl, &jr, &sense, &N, local_trans.data(), desc_A.data(),
          alphar.data(), alphai.data(), vl.data(), desc_A.data(), local_vr.data(), desc_A.data(),
          &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv, work.data(), &lwork, &info);

  if (info != 0) {
    std::cout << "dgeev error, info = " << info << std::endl;
  }
  auto dgeev_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dgeev_time = dgeev_end - dgeev_start;
  if (myrank == 0) std::cout << "  Eigenvalue solver time : " << dgeev_time.count() << std::endl;

  // copy eigenvectors back
  pdgemr2d_(&N, &N, local_vr.data(), &ia, &ja, desc_A.data(), vr.data(), &ib, &jb, desc_B.data(), &ictxt_blacs);


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

#if 0
  std::cout << "Adjusted eigenvalues" << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << std::setprecision(10) << i << " " << adjusted_ev[i] << std::endl;
  }
#endif

  std::vector<double>::iterator min_elem = std::min_element(adjusted_ev.begin(), adjusted_ev.end());
  int idx = std::distance(adjusted_ev.begin(), min_elem);
  *lowest_ev = alphar[idx];


  if (myrank == 0) {
    std::cout << "Index of min eigenvalue = " << idx << std::endl;
    std::cout << "Eigenvectors" << std::endl;
    for (int i = 0; i < N; i++) {
      double eval = vr[idx*N + i];
      scaled_eval[i] = eval/vr[idx*N];
      //std::cout << i << " " << eval << " " << scaled_eval[i] << std::endl;
    }
  }

}


// There are many variants of the linear method that adjust the Hamiltonian matrix
// to improve stability of the optimization.
// This follows the "one_shift_only" code in QMCFixedSampleLinearOptimizeBatched.cpp


int main(int argc, char **argv)
{

  MPI_Init(&argc, &argv);

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //std::string fname = "linear_matrices.h5";
  //std::string fname = "propane5k_linear_matrices.h5";
  std::string fname = "propane100_linear_matrices.h5";
  //std::string fname = "prop_three_body.h5";
  //std::string fname = "methane_linear_matrices.h5";

  int N;
  std::vector<double> hamiltonian_data;
  std::vector<double> overlap_data;
  double qmcpack_lowest_ev;
  std::vector<double> qmcpack_scaled_evec;

  if (myrank == 0) {

    if (argc > 1) {
      fname = argv[1];
    }

    std::cout << "Using file: " << fname << std::endl;

    hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      std::cout << "Unable to open: " << fname << std::endl;
      return 1;
    }

    std::chrono::system_clock::time_point read_start = std::chrono::system_clock::now();

    hid_t h1 = H5Dopen(file_id,"overlap",H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(h1);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    std::cout << " rank = " << rank << std::endl;

    std::vector<hsize_t> sizes(rank);
    H5Sget_simple_extent_dims(dataspace, sizes.data(), NULL);
    std::cout << " size = " << sizes[0] << std::endl;

    N = sizes[0];

    hsize_t matrix_size= sizes[0] * sizes[1];
    //std::vector<double> overlap_data(matrix_size);
    overlap_data.resize(matrix_size);
    H5Dread(h1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, overlap_data.data());
    std::cout <<"overlap = " << overlap_data[0] <<  " " << overlap_data[11] << std::endl;


    //std::vector<double> hamiltonian_data(matrix_size);
    hamiltonian_data.resize(matrix_size);
    hid_t h2 = H5Dopen(file_id,"Hamiltonian",H5P_DEFAULT);
    H5Dread(h2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hamiltonian_data.data());

    std::cout <<"ham = " << hamiltonian_data[0] <<  " " << hamiltonian_data[11] << std::endl;

    hid_t h3 = H5Dopen(file_id,"lowest_eigenvalue",H5P_DEFAULT);
    H5Dread(h3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &qmcpack_lowest_ev);

    //std::vector<double> qmcpack_scaled_evec(N);
    qmcpack_scaled_evec.resize(N);
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
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double lowest_ev;
  std::vector<double> scaled_evec(N);

  auto eig_start = std::chrono::system_clock::now();

  //do_inverse(overlap_data.data(), sizes[0]);
  //compute_eigenthings_generalized(N, overlap_data.data(), hamiltonian_data.data(), &lowest_ev, scaled_evec.data());
  //
  compute_eigenthings_other(N, overlap_data.data(), hamiltonian_data.data(), &lowest_ev, scaled_evec.data());
  auto eig_end = std::chrono::system_clock::now();
  std::chrono::duration<double> eig_time = eig_end - eig_start;
  if (myrank == 0) {
    std::cout << "Total generalized eigenvalue time: " << eig_time.count() << std::endl;
  }


  if (myrank == 0) {
    std::cout << std::setprecision(10) << "lowest ev: qmcpack, this, diff " << qmcpack_lowest_ev << " " << lowest_ev << " " << (qmcpack_lowest_ev - lowest_ev) << std::endl;

    double norm_sum = 0.0;
    for (int i = 0; i < N; i++) {
      //if (myrank == 0) {
      //  std::cout << i << " evec " << scaled_evec[i] << " " << qmcpack_scaled_evec[i] << std::endl;
      //}
      double diff = scaled_evec[i] - qmcpack_scaled_evec[i];
      norm_sum += diff*diff;
    }

    double norm = std::sqrt(norm_sum);
    std::cout << " norm diff scaled evec: " << norm << std::endl;
  }

  MPI_Finalize();
  return 0;
}
