
// Compute overlap and hamiltonian matrices from parameter derivative data
// stored in *.param_deriv.h5 files And optionally compare the results with the
// matrices stored in *.linear_matrices.h5

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <hdf5.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "matrix_wrapper.hpp"

std::vector<double> read_vector(hid_t file_id, const std::string &name,
                                std::vector<hsize_t> &sizes) {
  std::cout << " Reading vector " << name << std::endl;
  hid_t h1 = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(h1);
  int rank = H5Sget_simple_extent_ndims(dataspace);
  std::cout << " rank = " << rank << std::endl;

  // std::vector<hsize_t> sizes(rank);
  sizes.resize(rank);
  H5Sget_simple_extent_dims(dataspace, sizes.data(), NULL);
  std::cout << " size = " << sizes[0] << std::endl;

  int N = sizes[0];

  hsize_t vector_size = sizes[0];
  std::vector<double> vector(vector_size);
  H5Dread(h1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vector.data());

  return vector;
}

std::vector<double> read_matrix(hid_t file_id, const std::string &name,
                                std::vector<hsize_t> &sizes) {
  std::cout << " Reading matrix " << name << std::endl;
  hid_t h1 = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(h1);
  int rank = H5Sget_simple_extent_ndims(dataspace);
  std::cout << " rank = " << rank << std::endl;

  // std::vector<hsize_t> sizes(rank);
  sizes.resize(rank);
  H5Sget_simple_extent_dims(dataspace, sizes.data(), NULL);
  std::cout << " size = " << sizes[0] << " " << sizes[1] << std::endl;

  int N = sizes[0];

  hsize_t matrix_size = sizes[0] * sizes[1];
  std::vector<double> matrix(matrix_size);
  H5Dread(h1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix.data());

  return matrix;
}

using Matrix = Wrapper2D<double>;
using RealType = double;
using ValueType = double;
using Return_rt = double;
using Return_t = double;

enum FieldIndex_OPT {
  LOGPSI_FIXED = 0,
  LOGPSI_FREE = 1,
  ENERGY_TOT = 2,
  ENERGY_FIXED = 3,
  ENERGY_NEW = 4,
  REWEIGHT = 5
};

enum SumIndex_OPT {
  SUM_E_BARE = 0,
  SUM_ESQ_BARE,
  SUM_ABSE_BARE,
  SUM_E_WGT,
  SUM_ESQ_WGT,
  SUM_ABSE_WGT,
  SUM_WGT,
  SUM_WGTSQ,
  SUM_INDEX_SIZE
};

void fillOverlapHamiltonianMatricesBlocked(
    const int nsamples, const int numParams, std::vector<double> SumValue,
    const Matrix &RecordsOnNode_, const Matrix &DerivRecords_,
    const Matrix &HDerivRecords_, const double w_beta, Matrix &Left,
    Matrix &Right) {
  RealType b2;
  b2 = w_beta;
  const int rank_local_num_samples_ = nsamples;

  // Right               = 0.0;
  // Left                = 0.0;
  double curAvg_w = SumValue[SUM_E_WGT] / SumValue[SUM_WGT];
  Return_rt curAvg2_w = SumValue[SUM_ESQ_WGT] / SumValue[SUM_WGT];
  RealType V_avg = curAvg2_w - curAvg_w * curAvg_w;
  std::vector<Return_t> D_avg(numParams, 0.0);
  Return_rt wgtinv = 1.0 / SumValue[SUM_WGT];

  std::chrono::system_clock::time_point first_loop_start =
      std::chrono::system_clock::now();

  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
    Return_rt weight = RecordsOnNode_(iw, REWEIGHT) * wgtinv;
    for (int pm = 0; pm < numParams; pm++) {
      D_avg[pm] += DerivRecords_(iw, pm) * weight;
    }
  }
  std::chrono::system_clock::time_point first_loop_end =
      std::chrono::system_clock::now();
  std::chrono::duration<double> first_loop_time =
      first_loop_end - first_loop_start;
  std::cout << "  First loop in fill time: " << first_loop_time.count()
            << std::endl;

  // myComm->allreduce(D_avg);
  //
  // Block size for samples
  const int sbs = 32;
  // Block size or parameters
  const int pbs = 32;

  double local_left_data[pbs * pbs];
  double local_right_data[pbs * pbs];
  double local_left_vec1[pbs];
  double local_left_vec2[pbs];

#pragma omp parallel for collapse(3) private(                                  \
        local_left_data, local_right_data, local_left_vec1, local_left_vec2)
  // #pragma omp parallel for collapse(3)
  //  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
  for (int iw_b = 0; iw_b < rank_local_num_samples_; iw_b += sbs) {
    for (int pm_b = 0; pm_b < numParams; pm_b += pbs) {
      for (int pm2_b = 0; pm2_b < numParams; pm2_b += pbs) {

        Matrix localLeft(pbs, pbs, local_left_data);
        Matrix localRight(pbs, pbs, local_right_data);
        for (int i = 0; i < pbs; i++) {
          local_left_vec1[i] = 0.0;
          local_left_vec2[i] = 0.0;
        }
        for (int i = 0; i < pbs * pbs; i++) {
          local_left_data[i] = 0.0;
          local_right_data[i] = 0.0;
        }
        for (int iw = iw_b; iw < std::min(rank_local_num_samples_, iw_b + sbs);
             iw++) {
          Return_rt weight = RecordsOnNode_(iw, REWEIGHT) * wgtinv;
          Return_rt eloc_new = RecordsOnNode_(iw, ENERGY_NEW);
          const Matrix &Dsaved = DerivRecords_;
          const Matrix &HDsaved = HDerivRecords_;

          for (int pm = pm_b; pm < std::min(numParams, pm_b + pbs); pm++) {
            Return_t wfe =
                (HDsaved(iw, pm) + (Dsaved(iw, pm) - D_avg[pm]) * eloc_new) *
                weight;
            Return_t wfd = (Dsaved(iw, pm) - D_avg[pm]) * weight;
            Return_t vterm = HDsaved(iw, pm) * (eloc_new - curAvg_w) +
                             (Dsaved(iw, pm) - D_avg[pm]) * eloc_new *
                                 (eloc_new - RealType(2.0) * curAvg_w);
            if (pm2_b == 0) {
              //                 Variance
              // Left(0, pm + 1) += b2 * std::real(vterm) * weight;
              // local_left_vec1[pm - pm_b] += b2 * std::real(vterm) * weight;
              // Left(pm + 1, 0) += b2 * std::real(vterm) * weight;
              // local_left_vec2[pm - pm_b] += b2 * std::real(vterm) * weight;
              //                 Hamiltonian
              // Left(0, pm + 1) += (1 - b2) * std::real(wfe);
              local_left_vec1[pm - pm_b] += (1 - b2) * std::real(wfe);
              // Left(pm + 1, 0) += (1 - b2) * std::real(wfd) * eloc_new;
              local_left_vec2[pm - pm_b] +=
                  (1 - b2) * std::real(wfd) * eloc_new;
            }
            for (int pm2 = pm2_b; pm2 < std::min(numParams, pm2_b + pbs);
                 pm2++) {
              //                Hamiltonian
              // Left(pm + 1, pm2 + 1) +=
              localLeft(pm - pm_b, pm2 - pm2_b) +=
                  std::real((1 - b2) * std::conj(wfd) *
                            (HDsaved(iw, pm2) +
                             (Dsaved(iw, pm2) - D_avg[pm2]) * eloc_new));
              //                Overlap
              RealType ovlij =
                  std::real(std::conj(wfd) * (Dsaved(iw, pm2) - D_avg[pm2]));
              // Right(pm + 1, pm2 + 1) += ovlij;
              localRight(pm - pm_b, pm2 - pm2_b) += ovlij;
              //                Variance
              RealType varij =
                  weight *
                  std::real(
                      (HDsaved(iw, pm) -
                       RealType(2.0) * std::conj(Dsaved(iw, pm) - D_avg[pm]) *
                           eloc_new) *
                      (HDsaved(iw, pm2) - RealType(2.0) *
                                              (Dsaved(iw, pm2) - D_avg[pm2]) *
                                              eloc_new));
              // Left(pm + 1, pm2 + 1) += b2 * (varij + V_avg * ovlij);
              localLeft(pm - pm_b, pm2 - pm2_b) += b2 * (varij + V_avg * ovlij);
            }
          }
        }

#if 1
#pragma omp critical
        for (int pm = pm_b; pm < std::min(numParams, pm_b + pbs); pm++) {
          Left(0, pm + 1) += local_left_vec1[pm - pm_b];
          Left(pm + 1, 0) += local_left_vec2[pm - pm_b];
          for (int pm2 = pm2_b; pm2 < std::min(numParams, pm2_b + pbs); pm2++) {
            Left(pm + 1, pm2 + 1) += localLeft(pm - pm_b, pm2 - pm2_b);
            Right(pm + 1, pm2 + 1) += localRight(pm - pm_b, pm2 - pm2_b);
          }
        }
#endif
      }
    }
  }
  // myComm->allreduce(Right);
  // myComm->allreduce(Left);
  Left(0, 0) = (1 - b2) * curAvg_w + b2 * V_avg;
  Right(0, 0) = 1.0;
}

#ifdef USE_OFFLOAD_BLOCKED
void fillOverlapHamiltonianMatricesOffloadBlocked(
    const int nsamples, const int numParams, std::vector<double> SumValue,
    Matrix &RecordsOnNode_in, Matrix &DerivRecords_in,
    Matrix &HDerivRecords_in, const double w_beta, Matrix &Left_in,
    Matrix &Right_in) {
  RealType b2;
  b2 = w_beta;
  const int rank_local_num_samples_ = nsamples;

  // Right               = 0.0;
  // Left                = 0.0;
  double curAvg_w = SumValue[SUM_E_WGT] / SumValue[SUM_WGT];
  Return_rt curAvg2_w = SumValue[SUM_ESQ_WGT] / SumValue[SUM_WGT];
  RealType V_avg = curAvg2_w - curAvg_w * curAvg_w;
  std::vector<Return_t> D_avg(numParams, 0.0);
  Return_rt wgtinv = 1.0 / SumValue[SUM_WGT];

  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
    Return_rt weight = RecordsOnNode_in(iw, REWEIGHT) * wgtinv;
    for (int pm = 0; pm < numParams; pm++) {
      D_avg[pm] += DerivRecords_in(iw, pm) * weight;
    }
  }

  // myComm->allreduce(D_avg);
  // Offload blocked
  // Block size for samples
  const int sbs = 16;
  // Block size or parameters
  const int pbs = 16;

  double local_left_data[pbs * pbs];
  double local_right_data[pbs * pbs];
  double local_left_vec1[pbs];
  double local_left_vec2[pbs];

  Return_rt* Left_ptr  = Left_in.data();
  Return_rt* Right_ptr = Right_in.data();
  const int out_size   = (numParams + 1) * (numParams + 1);
  const int data_size  = numParams * rank_local_num_samples_;

  Return_rt* Dsaved_ptr  = DerivRecords_in.data();
  Return_rt* HDsaved_ptr = HDerivRecords_in.data();
  Return_rt* Records_ptr = RecordsOnNode_in.data();

  Return_rt* D_avg1 = D_avg.data();


#pragma omp target data map(tofrom : Left_ptr[ : out_size], Right_ptr[ : out_size])         \
    map(to : Records_ptr[ : rank_local_num_samples_ * SUM_INDEX_SIZE], D_avg1[ : numParams]) \
    map(to : Dsaved_ptr[ : data_size], HDsaved_ptr[ : data_size])
  {


//#pragma omp target teams distribute collapse(3) private(                                  \
#pragma omp parallel for collapse(3) private(                                  \
        local_left_data, local_right_data, local_left_vec1, local_left_vec2) \
     map(tofrom : Left_ptr[ : out_size], Right_ptr[ : out_size])         \
    map(to : Records_ptr[ : rank_local_num_samples_ * SUM_INDEX_SIZE], D_avg1[ : numParams]) \
    map(to : Dsaved_ptr[ : data_size], HDsaved_ptr[ : data_size])
  // #pragma omp parallel for collapse(3)
  //  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
  for (int iw_b = 0; iw_b < rank_local_num_samples_; iw_b += sbs) {
    for (int pm_b = 0; pm_b < numParams; pm_b += pbs) {
      for (int pm2_b = 0; pm2_b < numParams; pm2_b += pbs) {
        Matrix Left(numParams+1, numParams+1, Left_ptr);
        Matrix Right(numParams+1, numParams+1, Right_ptr);
        Matrix RecordsOnNode_(rank_local_num_samples_, REWEIGHT, Records_ptr);
        Matrix DerivRecords_(rank_local_num_samples_, numParams, Dsaved_ptr);
        Matrix HDerivRecords_(rank_local_num_samples_, numParams, HDsaved_ptr);

        Matrix localLeft(pbs, pbs, local_left_data);
        Matrix localRight(pbs, pbs, local_right_data);
        for (int i = 0; i < pbs; i++) {
          local_left_vec1[i] = 0.0;
          local_left_vec2[i] = 0.0;
        }
        for (int i = 0; i < pbs * pbs; i++) {
          local_left_data[i] = 0.0;
          local_right_data[i] = 0.0;
        }
//#pragma omp parallel for collapse(2)
//#pragma omp parallel for
        for (int iw = iw_b; iw < std::min(rank_local_num_samples_, iw_b + sbs);
             iw++) {
          Return_rt weight = RecordsOnNode_(iw, REWEIGHT) * wgtinv;
          Return_rt eloc_new = RecordsOnNode_(iw, ENERGY_NEW);
          const Matrix &Dsaved = DerivRecords_;
          const Matrix &HDsaved = HDerivRecords_;

          for (int pm = pm_b; pm < std::min(numParams, pm_b + pbs); pm++) {
            Return_t wfe =
                (HDsaved(iw, pm) + (Dsaved(iw, pm) - D_avg1[pm]) * eloc_new) *
                weight;
            Return_t wfd = (Dsaved(iw, pm) - D_avg1[pm]) * weight;
            Return_t vterm = HDsaved(iw, pm) * (eloc_new - curAvg_w) +
                             (Dsaved(iw, pm) - D_avg1[pm]) * eloc_new *
                                 (eloc_new - RealType(2.0) * curAvg_w);
            if (pm2_b == 0) {
              //                 Variance
              // Left(0, pm + 1) += b2 * std::real(vterm) * weight;
              // local_left_vec1[pm - pm_b] += b2 * std::real(vterm) * weight;
              // Left(pm + 1, 0) += b2 * std::real(vterm) * weight;
              // local_left_vec2[pm - pm_b] += b2 * std::real(vterm) * weight;
              //                 Hamiltonian
              // Left(0, pm + 1) += (1 - b2) * std::real(wfe);
              local_left_vec1[pm - pm_b] += (1 - b2) * std::real(wfe);
              // Left(pm + 1, 0) += (1 - b2) * std::real(wfd) * eloc_new;
              local_left_vec2[pm - pm_b] +=
                  (1 - b2) * std::real(wfd) * eloc_new;
            }
            for (int pm2 = pm2_b; pm2 < std::min(numParams, pm2_b + pbs);
                 pm2++) {
              //                Hamiltonian
              // Left(pm + 1, pm2 + 1) +=
              localLeft(pm - pm_b, pm2 - pm2_b) +=
                  std::real((1 - b2) * std::conj(wfd) *
                            (HDsaved(iw, pm2) +
                             (Dsaved(iw, pm2) - D_avg1[pm2]) * eloc_new));
              //                Overlap
              RealType ovlij =
                  std::real(std::conj(wfd) * (Dsaved(iw, pm2) - D_avg1[pm2]));
              // Right(pm + 1, pm2 + 1) += ovlij;
              localRight(pm - pm_b, pm2 - pm2_b) += ovlij;
              //                Variance
              RealType varij =
                  weight *
                  std::real(
                      (HDsaved(iw, pm) -
                       RealType(2.0) * std::conj(Dsaved(iw, pm) - D_avg1[pm]) *
                           eloc_new) *
                      (HDsaved(iw, pm2) - RealType(2.0) *
                                              (Dsaved(iw, pm2) - D_avg1[pm2]) *
                                              eloc_new));
              // Left(pm + 1, pm2 + 1) += b2 * (varij + V_avg * ovlij);
              localLeft(pm - pm_b, pm2 - pm2_b) += b2 * (varij + V_avg * ovlij);
            }
          }
        }

#if 1
// How do we do a reduction across teams?
#pragma omp critical
//#pragma omp parallel for
        for (int pm = pm_b; pm < std::min(numParams, pm_b + pbs); pm++) {
          Left(0, pm + 1) += local_left_vec1[pm - pm_b];
          Left(pm + 1, 0) += local_left_vec2[pm - pm_b];
          for (int pm2 = pm2_b; pm2 < std::min(numParams, pm2_b + pbs); pm2++) {
            Left(pm + 1, pm2 + 1) += localLeft(pm - pm_b, pm2 - pm2_b);
            Right(pm + 1, pm2 + 1) += localRight(pm - pm_b, pm2 - pm2_b);
          }
        }
#endif
      }
    }
  }

  } // target data
  // myComm->allreduce(Right);
  // myComm->allreduce(Left);
  Left_in(0, 0) = (1 - b2) * curAvg_w + b2 * V_avg;
  Right_in(0, 0) = 1.0;
}
#endif

#ifdef USE_OFFLOAD
void fillOverlapHamiltonianMatricesOffload(const int nsamples, const int numParams,
                                    std::vector<double> SumValue,
                                    Matrix &RecordsOnNode_,
                                    Matrix &DerivRecords_,
                                    Matrix &HDerivRecords_,
                                    const double w_beta, Matrix &Left,
                                    Matrix &Right) {
  RealType b2;
  b2 = w_beta;


  const int rank_local_num_samples_ = nsamples;

  double curAvg_w = SumValue[SUM_E_WGT] / SumValue[SUM_WGT];
  Return_rt curAvg2_w = SumValue[SUM_ESQ_WGT] / SumValue[SUM_WGT];
  RealType V_avg = curAvg2_w - curAvg_w * curAvg_w;
  std::vector<Return_t> D_avg(numParams, 0.0);
  Return_rt wgtinv = 1.0 / SumValue[SUM_WGT];

  std::chrono::system_clock::time_point first_loop_start =
      std::chrono::system_clock::now();

  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
    Return_rt weight = RecordsOnNode_(iw, REWEIGHT) * wgtinv;
    for (int pm = 0; pm < numParams; pm++) {
      D_avg[pm] += DerivRecords_(iw, pm) * weight;
    }
  }

  //const Matrix<Return_t>& Dsaved = DerivRecords_;
  //const Matrix<Return_t>& HDsaved = HDerivRecords_;
  std::vector<double> Dsaved_data(numParams*rank_local_num_samples_);
  std::vector<double> HDsaved_data(numParams*rank_local_num_samples_);
  Matrix Dsaved(numParams, rank_local_num_samples_,Dsaved_data.data());
  Matrix HDsaved(numParams, rank_local_num_samples_,HDsaved_data.data());
  for (int iw = 0; iw < rank_local_num_samples_; iw++)
    for (int pm = 0; pm < numParams; pm++)
    {
      Dsaved(pm, iw)  = DerivRecords_(iw, pm);
      HDsaved(pm, iw) = HDerivRecords_(iw, pm);
    }

  Return_rt* Left_ptr  = Left.data();
  Return_rt* Right_ptr = Right.data();
  const int out_size   = (numParams + 1) * (numParams + 1);
  const int data_size  = numParams * rank_local_num_samples_;

  Return_rt* Dsaved_ptr  = Dsaved.data();
  Return_rt* HDsaved_ptr = HDsaved.data();
  Return_rt* Records_ptr = RecordsOnNode_.data();

  const uint64_t data_bytes = data_size * sizeof(Return_t);
  const uint64_t out_bytes  = out_size * sizeof(Return_t);
  std::cout << "Data matrices size, each = " << data_bytes / 1000 << " K" << " total = " << 2 * data_bytes / 1e6 << " M"
            << std::endl;
  std::cout << "Output matrices size, each = " << out_bytes / 1000 << " K" << " total = " << 2 * out_bytes / 1e6 << " M"
            << std::endl;

  Return_rt* D_avg1 = D_avg.data();

#pragma omp target data map(tofrom : Left_ptr[ : out_size], Right_ptr[ : out_size])         \
    map(to : Records_ptr[ : rank_local_num_samples_ * SUM_INDEX_SIZE], D_avg1[ : numParams]) \
    map(to : Dsaved_ptr[ : data_size], HDsaved_ptr[ : data_size])
  {
#pragma omp target teams distribute
    for (int pm = 0; pm < numParams; pm++)
    {
      //Matrix<Return_rt> Records1;
      //Records1.attachReference(Records_ptr, rank_local_num_samples_, SUM_INDEX_SIZE);
      Matrix Records1(rank_local_num_samples_, SUM_INDEX_SIZE, Records_ptr);

      Matrix Left1(numParams+1, numParams+1, Left_ptr);;
      //Left1.attachReference(Left_ptr, numParams + 1, numParams + 1);
      Matrix Right1(numParams+1, numParams+1, Right_ptr);;
      //Right1.attachReference(Right_ptr, numParams + 1, numParams + 1);

      Matrix Dsaved1(numParams, rank_local_num_samples_, Dsaved_ptr);
      //Dsaved1.attachReference(Dsaved_ptr, numParams, rank_local_num_samples_);
      Matrix HDsaved1(numParams, rank_local_num_samples_, HDsaved_ptr);
      //HDsaved1.attachReference(HDsaved_ptr, numParams, rank_local_num_samples_);

      Return_rt left1(0.0);
      Return_rt left3(0.0);
      Return_rt left4(0.0);

#pragma omp parallel for reduction(+ : left1, left3, left4)
      for (int iw = 0; iw < rank_local_num_samples_; iw++)
      {
        Return_rt weight   = Records1(iw, REWEIGHT) * wgtinv;
        Return_rt eloc_new = Records1(iw, ENERGY_NEW);
        Return_t wfe       = (HDsaved1(pm, iw) + (Dsaved1(pm, iw) - D_avg1[pm]) * eloc_new) * weight;
        Return_t wfd       = (Dsaved1(pm, iw) - D_avg1[pm]) * weight;
        Return_t vterm     = HDsaved1(pm, iw) * (eloc_new - curAvg_w) +
            (Dsaved1(pm, iw) - D_avg1[pm]) * eloc_new * (eloc_new - RealType(2.0) * curAvg_w);
        //                 Variance
        //Left(0, pm + 1) += b2 * std::real(vterm) * weight;
        //Left(pm + 1, 0) += b2 * std::real(vterm) * weight;
        left1 += b2 * std::real(vterm) * weight;
        //                 Hamiltonian
        //Left(0, pm + 1) += (1 - b2) * std::real(wfe);
        //Left(pm + 1, 0) += (1 - b2) * std::real(wfd) * eloc_new;
        left3 += (1 - b2) * std::real(wfe);
        left4 += (1 - b2) * std::real(wfd) * eloc_new;
      }
      Left1(0, pm + 1) = left1 + left3;
      Left1(pm + 1, 0) = left1 + left4;
    }
#pragma omp target teams distribute
    for (int pm = 0; pm < numParams; pm++)
    {
      Matrix Records1(rank_local_num_samples_, SUM_INDEX_SIZE, Records_ptr);

      Matrix Left1(numParams+1, numParams+1, Left_ptr);;
      Matrix Right1(numParams+1, numParams+1, Right_ptr);;

      Matrix Dsaved1(numParams, rank_local_num_samples_, Dsaved_ptr);
      Matrix HDsaved1(numParams, rank_local_num_samples_, HDsaved_ptr);

#if 0
      Matrix<Return_rt> Records1;
      Records1.attachReference(Records_ptr, rank_local_num_samples_, SUM_INDEX_SIZE);

      Matrix<Return_rt> Left1;
      Left1.attachReference(Left_ptr, numParams + 1, numParams + 1);
      Matrix<Return_rt> Right1;
      Right1.attachReference(Right_ptr, numParams + 1, numParams + 1);

      Matrix<Return_rt> Dsaved1;
      Dsaved1.attachReference(Dsaved_ptr, numParams, rank_local_num_samples_);
      Matrix<Return_rt> HDsaved1;
      HDsaved1.attachReference(HDsaved_ptr, numParams, rank_local_num_samples_);
#endif

      for (int pm2 = 0; pm2 < numParams; pm2++)
      {
        Return_rt left1(0.0);
        Return_rt left2(0.0);
        Return_rt right1(0.0);
#pragma omp parallel for reduction(+ : left1, left2, right1)
        for (int iw = 0; iw < rank_local_num_samples_; iw++)
        {
          Return_rt weight   = Records1(iw, REWEIGHT) * wgtinv;
          Return_rt eloc_new = Records1(iw, ENERGY_NEW);
          Return_t wfd       = (Dsaved1(pm, iw) - D_avg1[pm]) * weight;
          //                Hamiltonian
          //Left(pm + 1, pm2 + 1) +=
          left1 +=
              std::real((1 - b2) * std::conj(wfd) * (HDsaved1(pm2, iw) + (Dsaved1(pm2, iw) - D_avg1[pm2]) * eloc_new));
          //                Overlap
          RealType ovlij = std::real(std::conj(wfd) * (Dsaved1(pm2, iw) - D_avg1[pm2]));
          //Right(pm + 1, pm2 + 1) += ovlij;
          right1 += ovlij;
          //                Variance
          RealType varij = weight *
              std::real((HDsaved1(pm, iw) - RealType(2.0) * std::conj(Dsaved1(pm, iw) - D_avg1[pm]) * eloc_new) *
                        (HDsaved1(pm2, iw) - RealType(2.0) * (Dsaved1(pm2, iw) - D_avg1[pm2]) * eloc_new));
          //Left(pm + 1, pm2 + 1) += b2 * (varij + V_avg * ovlij);
          left2 += b2 * (varij + V_avg * ovlij);
        }
        Left1(pm + 1, pm2 + 1)  = left1 + left2;
        Right1(pm + 1, pm2 + 1) = right1;
      }
    }
  }




  Left(0, 0)  = (1 - b2) * curAvg_w + b2 * V_avg;
  Right(0, 0) = 1.0;



}
#endif

void fillOverlapHamiltonianMatrices(const int nsamples, const int numParams,
                                    std::vector<double> SumValue,
                                    const Matrix &RecordsOnNode_,
                                    const Matrix &DerivRecords_,
                                    const Matrix &HDerivRecords_,
                                    const double w_beta, Matrix &Left,
                                    Matrix &Right) {
  RealType b2;
  b2 = w_beta;
  const int rank_local_num_samples_ = nsamples;

  // Right               = 0.0;
  // Left                = 0.0;
  double curAvg_w = SumValue[SUM_E_WGT] / SumValue[SUM_WGT];
  Return_rt curAvg2_w = SumValue[SUM_ESQ_WGT] / SumValue[SUM_WGT];
  RealType V_avg = curAvg2_w - curAvg_w * curAvg_w;
  std::vector<Return_t> D_avg(numParams, 0.0);
  Return_rt wgtinv = 1.0 / SumValue[SUM_WGT];

  std::chrono::system_clock::time_point first_loop_start =
      std::chrono::system_clock::now();

  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
    Return_rt weight = RecordsOnNode_(iw, REWEIGHT) * wgtinv;
    for (int pm = 0; pm < numParams; pm++) {
      D_avg[pm] += DerivRecords_(iw, pm) * weight;
    }
  }
  std::chrono::system_clock::time_point first_loop_end =
      std::chrono::system_clock::now();
  std::chrono::duration<double> first_loop_time =
      first_loop_end - first_loop_start;
  std::cout << "  First loop in fill time: " << first_loop_time.count()
            << std::endl;

  // myComm->allreduce(D_avg);

  for (int iw = 0; iw < rank_local_num_samples_; iw++) {
    Return_rt weight = RecordsOnNode_(iw, REWEIGHT) * wgtinv;
    Return_rt eloc_new = RecordsOnNode_(iw, ENERGY_NEW);
    const Matrix &Dsaved = DerivRecords_;
    const Matrix &HDsaved = HDerivRecords_;

#pragma omp parallel for
    for (int pm = 0; pm < numParams; pm++) {
      Return_t wfe =
          (HDsaved(iw, pm) + (Dsaved(iw, pm) - D_avg[pm]) * eloc_new) * weight;
      Return_t wfd = (Dsaved(iw, pm) - D_avg[pm]) * weight;
      Return_t vterm = HDsaved(iw, pm) * (eloc_new - curAvg_w) +
                       (Dsaved(iw, pm) - D_avg[pm]) * eloc_new *
                           (eloc_new - RealType(2.0) * curAvg_w);
      //                 Variance
      Left(0, pm + 1) += b2 * std::real(vterm) * weight;
      Left(pm + 1, 0) += b2 * std::real(vterm) * weight;
      //                 Hamiltonian
      Left(0, pm + 1) += (1 - b2) * std::real(wfe);
      Left(pm + 1, 0) += (1 - b2) * std::real(wfd) * eloc_new;
      for (int pm2 = 0; pm2 < numParams; pm2++) {
        //                Hamiltonian
        Left(pm + 1, pm2 + 1) += std::real(
            (1 - b2) * std::conj(wfd) *
            (HDsaved(iw, pm2) + (Dsaved(iw, pm2) - D_avg[pm2]) * eloc_new));
        //                Overlap
        RealType ovlij =
            std::real(std::conj(wfd) * (Dsaved(iw, pm2) - D_avg[pm2]));
        Right(pm + 1, pm2 + 1) += ovlij;
        //                Variance
        RealType varij =
            weight *
            std::real(
                (HDsaved(iw, pm) - RealType(2.0) *
                                       std::conj(Dsaved(iw, pm) - D_avg[pm]) *
                                       eloc_new) *
                (HDsaved(iw, pm2) -
                 RealType(2.0) * (Dsaved(iw, pm2) - D_avg[pm2]) * eloc_new));
        Left(pm + 1, pm2 + 1) += b2 * (varij + V_avg * ovlij);
      }
    }
  }
  // myComm->allreduce(Right);
  // myComm->allreduce(Left);
  Left(0, 0) = (1 - b2) * curAvg_w + b2 * V_avg;
  Right(0, 0) = 1.0;
}

void read_matrices_file(const std::string &fname,
                        std::vector<double> &overlap_data,
                        std::vector<double> &hamiltonian_data) {

  std::cout << "Reading matrices file : " << fname << std::endl;
  hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cout << "Unable to open: " << fname << std::endl;
  }
  hid_t h1 = H5Dopen(file_id, "overlap", H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(h1);
  int rank = H5Sget_simple_extent_ndims(dataspace);
  std::cout << " rank = " << rank << std::endl;

  std::vector<hsize_t> sizes(rank);
  H5Sget_simple_extent_dims(dataspace, sizes.data(), NULL);
  std::cout << " size = " << sizes[0] << std::endl;

  int N = sizes[0];

  hsize_t matrix_size = sizes[0] * sizes[1];
  overlap_data.resize(matrix_size);
  H5Dread(h1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          overlap_data.data());
  std::cout << "overlap = " << overlap_data[0] << " " << overlap_data[11]
            << std::endl;

  hamiltonian_data.resize(matrix_size);
  hid_t h2 = H5Dopen(file_id, "Hamiltonian", H5P_DEFAULT);
  H5Dread(h2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          hamiltonian_data.data());

  std::cout << "ham = " << hamiltonian_data[0] << " " << hamiltonian_data[11]
            << std::endl;
}

int main(int argc, char **argv) {

  std::string fname = "opt.s000.param_deriv.h5";

  if (argc > 1) {
    fname = argv[1];
  }

  std::string fname_matrices;
  if (argc > 2) {
    fname_matrices = argv[2];
  }

  std::cout << "Using file: " << fname << std::endl;

  hid_t file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cout << "Unable to open: " << fname << std::endl;
    return 1;
  }

  std::chrono::system_clock::time_point read_start =
      std::chrono::system_clock::now();

  std::vector<hsize_t> sizes;
  auto SumValue = read_vector(file_id, "SumValue", sizes);
  assert(sizes.size() == 1);

  auto RecordsOnNodeData = read_matrix(file_id, "RecordsOnNode", sizes);
  assert(sizes.size() == 2);
  int nsamples = sizes[0];
  int nfixed = sizes[1];
  Wrapper2D<double> RecordsOnNode(nsamples, nfixed, RecordsOnNodeData.data());

  auto DerivRecordsData = read_matrix(file_id, "DerivRecords", sizes);
  assert(sizes.size() == 2);
  assert(sizes[0] == nsamples);
  int nparam = sizes[1];
  Wrapper2D<double> DerivRecords(nsamples, nparam, DerivRecordsData.data());

  auto HDerivRecordsData = read_matrix(file_id, "HDerivRecords", sizes);
  assert(sizes.size() == 2);
  std::cout << "HDerivRecords size = " << sizes[0] << " " << sizes[1]
            << std::endl;
  assert(sizes[0] == nsamples);
  assert(sizes[1] == nparam);
  Wrapper2D<double> HDerivRecords(nsamples, nparam, HDerivRecordsData.data());

  double w_beta;
  hid_t h6 = H5Dopen(file_id, "w_beta", H5P_DEFAULT);
  H5Dread(h6, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &w_beta);

  H5Fclose(file_id);

  std::chrono::system_clock::time_point read_end =
      std::chrono::system_clock::now();
  std::chrono::duration<double> read_time = read_end - read_start;
  std::cout << "File read time: " << read_time.count() << std::endl;

  std::vector<double> gold_overlap_data;
  std::vector<double> gold_hamiltonian_data;
  bool do_compare = false;
  if (fname_matrices.size() > 0) {
    read_matrices_file(fname_matrices, gold_overlap_data,
                       gold_hamiltonian_data);
    do_compare = true;
  }

  std::vector<double> overlapData((nparam + 1) * (nparam + 1));
  Matrix overlap(nparam + 1, nparam + 1, overlapData.data());
  std::vector<double> hamData((nparam + 1) * (nparam + 1));
  Matrix ham(nparam + 1, nparam + 1, hamData.data());

  std::chrono::system_clock::time_point fill_start =
      std::chrono::system_clock::now();

#if 0
  fillOverlapHamiltonianMatrices(nsamples, nparam, SumValue, RecordsOnNode,
                                 DerivRecords, HDerivRecords, w_beta, ham,
                                 overlap);
#endif
#if 1
  fillOverlapHamiltonianMatricesBlocked(nsamples, nparam, SumValue,
                                        RecordsOnNode, DerivRecords,
                                        HDerivRecords, w_beta, ham, overlap);
#endif
#ifdef USE_OFFLOAD
  fillOverlapHamiltonianMatricesOffload(nsamples, nparam, SumValue,
                                        RecordsOnNode, DerivRecords,
                                        HDerivRecords, w_beta, ham, overlap);
#endif
#ifdef USE_OFFLOAD_BLOCKED
  fillOverlapHamiltonianMatricesOffloadBlocked(nsamples, nparam, SumValue,
                                        RecordsOnNode, DerivRecords,
                                        HDerivRecords, w_beta, ham, overlap);
#endif

  std::chrono::system_clock::time_point fill_end =
      std::chrono::system_clock::now();
  std::chrono::duration<double> fill_time = fill_end - fill_start;
  std::cout << "Fill time: " << fill_time.count() << std::endl;

  if (do_compare) {
    std::cout << "Checking matrices" << std::endl;
    Matrix gold_overlap(nparam + 1, nparam + 1, gold_overlap_data.data());
    Matrix gold_ham(nparam + 1, nparam + 1, gold_hamiltonian_data.data());

    assert(gold_overlap.rows() == overlap.rows());
    assert(gold_overlap.cols() == overlap.cols());
    assert(gold_ham.rows() == ham.rows());
    assert(gold_ham.cols() == ham.cols());

    double tol = 1.0e-5;
    int nerror = 0;
    for (int i = 0; i < nparam; i++) {
      for (int j = 0; j < nparam; j++) {
        double diff = gold_overlap(i, j) - overlap(i, j);
        if (std::abs(diff) > tol) {
          nerror++;
          if (nerror < 10) {
            std::cout << "Overlap difference " << i << " " << j << " "
                      << gold_overlap(i, j) << " " << overlap(i, j)
                      << std::endl;
          }
        }
      }
    }
    if (nerror == 0)
      std::cout << "Overlap passed" << std::endl;
    else
      std::cout << "Errors in overlap = " << nerror << std::endl;

    nerror = 0;
    for (int i = 0; i < nparam; i++) {
      for (int j = 0; j < nparam; j++) {
        double diff = gold_ham(i, j) - ham(i, j);
        if (std::abs(diff) > tol) {
          nerror++;
          if (nerror < 10) {
            std::cout << "Hamiltonian difference " << i << " " << j << " "
                      << gold_ham(i, j) << " " << ham(i, j) << std::endl;
          }
        }
      }
    }
    if (nerror == 0)
      std::cout << "Hamiltonian passed" << std::endl;
    else
      std::cout << "Errors in hamiltonian = " << nerror << std::endl;
  }

  return 0;
}
