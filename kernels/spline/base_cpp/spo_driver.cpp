
#include <vector>
#include <random>
#include <sys/time.h>
#include <stddef.h>
#include <cmath>
#include <array>

#include <MultiBsplineRef.hpp>
#include <MultiBspline.hpp>
#include <MultiBsplineData.hpp>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

// This is the 3D spline code from QMCPACK and miniQMC extracted as a standalone kernel

// The spline coefficient data can be large and is read-only during spline evaluation.
// There are multiple values (nsplines) at each point in 3D space.
// The main evaluation routines
//  - evaluate_v - returns a vector (nsplines) of values
//  - evaluate_vgl - returns a vector (nsplines) of values, gradients, and Laplacians


using namespace qmcplusplus;

inline double cpu_clock()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec+(1.e-6)*tv.tv_usec;
}

//typedef double RealType;
typedef float RealType;

void move_to_box(std::array<RealType, 3>& pos)
{
    RealType grid_start = 0.0;
    RealType grid_end = 1.0;
    RealType box_len = grid_end - grid_start;
    for (int i = 0; i < 3; i++) {
      RealType loc = pos[i] - grid_start;
      RealType ipart;
      pos[i] = std::modf(loc/box_len, &ipart) * box_len +  grid_start;
    }
}


void setup_spline_data(bspline_traits<RealType, 3>::SplineType &spline_data, int grid_num, int nspline)
{
  int coef_num = grid_num + 3;
  double grid_start = 0.0;
  double grid_end = 1.0;
  spline_data.num_splines = nspline;
  double delta = (grid_start - grid_end)/grid_num;
  double delta_inv = 1.0/delta;

  spline_data.x_grid.start = grid_start;
  spline_data.x_grid.end = grid_end;
  spline_data.x_grid.num = grid_num;
  spline_data.x_grid.delta = delta;
  spline_data.x_grid.delta_inv = delta_inv;
  spline_data.x_stride = coef_num * coef_num * nspline;

  spline_data.y_grid.start = grid_start;
  spline_data.y_grid.end = grid_end;
  spline_data.y_grid.num = grid_num;
  spline_data.y_grid.delta = delta;
  spline_data.y_grid.delta_inv = delta_inv;
  spline_data.y_stride = coef_num * nspline;

  spline_data.z_grid.start = grid_start;
  spline_data.z_grid.end = grid_end;
  spline_data.z_grid.num = grid_num;
  spline_data.z_grid.delta = delta;
  spline_data.z_grid.delta_inv = delta_inv;
  spline_data.z_stride = nspline;

  int coef_size = coef_num * coef_num * coef_num * nspline;
  spline_data.coefs = (RealType *)malloc(sizeof(RealType) * coef_size);

  size_t coef_bytes = coef_size * sizeof(RealType);
  double coef_mb = coef_bytes/(1024.0)/1024; // in MB
  std::cout << " Coefficient size (MB) = " << coef_mb << std::endl;

  std::mt19937 e1(1);
  std::uniform_real_distribution<RealType> uniform(0.0, 1.0);
  for (int i = 0; i < coef_size; i++) {
    spline_data.coefs[i] = uniform(e1);
  }
}

int main(int argc, char **argv)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm m_world = MPI_COMM_WORLD;
#endif

  int nx = 37;
  int ny = nx;
  int nz = nx;
  int nspline = 192;


  std::vector<RealType> psi, psi_ref;
  psi.resize(nspline);
  psi_ref.resize(nspline);

  std::vector<RealType> grad, grad_ref;
  grad.resize(3*nspline);
  grad_ref.resize(3*nspline);

  std::vector<RealType> lapl, lapl_ref;
  lapl.resize(3*nspline);
  lapl_ref.resize(3*nspline);

  std::array<RealType, 3> pos = {0.0, 0.0, 0.0};

  // Increment for repeat calls.  Use 0 for repeated access to the same coefficients
  //std::array<RealType, 3> inc = {0.0, 0.0, 0.0};
  std::array<RealType, 3> inc = {0.11, 0.37, 0.07};

  bspline_traits<RealType, 3>::SplineType spline_data;

  setup_spline_data(spline_data, nx, nspline);

  miniqmcreference::MultiBsplineRef<RealType> spline_ref;

  int nrepeat = 100;

  int coef_bytes_loaded = nspline * 64 * sizeof(RealType);

  // Ops numbers from hand-counting.
  int vgl_adds = 3 + 12*3*3 + ((3*3 + 7)*nspline + 6)*16 + nspline*2;
  int vgl_muls = 12*3*3 + ((4*3 + 7)*nspline)*16 + 3 + nspline*6;
  int vgl_flop = vgl_adds + vgl_muls;

  double ref_start = cpu_clock();
  for (int i = 0; i < nrepeat; i++) {
    //spline_ref.evaluate_v(&spline_data, pos[0], pos[1], pos[2], psi_ref.data(), nspline);
    spline_ref.evaluate_vgl(&spline_data, pos[0], pos[1], pos[2], psi_ref.data(), grad_ref.data(), lapl_ref.data(), nspline);
    for (int j = 0; j < 3; j++) {
      pos[j] += inc[j];
      move_to_box(pos);
    }
  }

  double ref_end = cpu_clock();

  std::cout << "ref spo psi[0] = " << psi_ref[0] << std::endl;
  std::cout << "ref spo psi[1] = " << psi_ref[1] << std::endl;
  double ref_time = (ref_end - ref_start)/nrepeat;
  std::cout << " ref time = " << ref_time << std::endl;
  double coef_bw = coef_bytes_loaded / ref_time;
  std::cout << " ref coef BW = " << coef_bw/1.0e9 << " (GB/s)" << std::endl;
  std::cout << " ref splines / sec = " << nspline/ref_time << std::endl;
  std::cout << " ref GFLOPS = " << vgl_flop/ref_time/1.0e9 << std::endl;

  std::cout << std::endl;


  MultiBspline<RealType> spline_new;
  double start = cpu_clock();
  for (int i = 0; i < nrepeat; i++) {
    //spline_new.evaluate_v(&spline_data, pos[0], pos[1], pos[2], psi.data(), nspline);
    spline_new.evaluate_vgl(&spline_data, pos[0], pos[1], pos[2], psi.data(), grad.data(), lapl.data(), nspline);
    for (int j = 0; j < 3; j++) {
      pos[j] += inc[j];
      move_to_box(pos);
    }
  }
  double end = cpu_clock();

  std::cout << "new spo psi[0] = " << psi[0] << std::endl;
  std::cout << "new spo psi[1] = " << psi[1] << std::endl;
  double new_time =  (end - start)/nrepeat;
  std::cout << " time = " << new_time << std::endl;
  double new_coef_bw = coef_bytes_loaded / new_time;
  std::cout << " coef BW = " << new_coef_bw/1.0e9 << " (GB/s)" << std::endl;
  std::cout << " splines / sec = " << nspline/new_time << std::endl;
  std::cout << " GFLOPS = " << vgl_flop/new_time/1.0e9 << std::endl;


#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
};
