
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#ifdef USE_FMT_LIBRARY
#include <fmt/core.h>
#endif

// Use OpenMP offload to target accelerators

const double pi = M_PI;

const int ndim = 6;

// Define the integrand

double mag(double r[3])
{
  return std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
}

double orb(double R[3])
{
  double r = mag(R);
  double Z = 2.0;
  double norm = 0.5/std::sqrt(pi);
  return norm*std::exp(-Z*r);
}

double jastrow(double r12, double B)
{
  double A = 0.5;
  return std::exp(A*r12/(1.0 + B*r12) - A/B);
}

double psi(double r1[3], double r2[3], double B)
{
  double o1 = orb(r1);
  double o2 = orb(r2);

  double r12[3];
  r12[0] = r2[0] - r1[0];
  r12[1] = r2[1] - r1[1];
  r12[2] = r2[2] - r1[2];

  double r12_mag = mag(r12);
  double j = jastrow(r12_mag, B);

  return o1*o2*j;
}

double psi_fn(double xx[ndim])
{
  double r1[3];
  double r2[3];

  r1[0] = xx[0];
  r1[1] = xx[1];
  r1[2] = xx[2];

  r2[0] = xx[3];
  r2[1] = xx[4];
  r2[2] = xx[5];

  double B = 0.1;
  double p = psi(r1, r2, B);
  return p*p;
}

// https://stackoverflow.com/questions/46782444/how-to-convert-a-linear-index-to-subscripts-with-support-for-negative-strides
void ind2sub(int idx, const int* shape, int* indices)
{
    for (int i = 0; i < ndim; i++)
    {
        int s = idx % shape[i];
        idx -= s;
        idx /= shape[i];
        indices[i] = s;
    }
}

uint64_t get_npts(int ndim, int n)
{
    return std::pow(n, ndim);
}

const double L = 0.5;

double transform_cc(double t)
{
    //return L/std::tan(t);

    return L*std::cos(t)/std::sin(t);
}

double jacobian_cc(double t)
{
    double s = std::sin(t);
    return L/(s*s);
}

double trapn_cc(int n)
{
    double h = pi/(n+1);

    double total = 0.0;

    int npts = std::pow(n, ndim);
    int indices[ndim];
    int nn[ndim];
    for (int i = 0; i < ndim; i++) {
        nn[i] = n;
    }

    double start = omp_get_wtime();

#pragma omp target parallel for reduction(+:total)
    for (int i = 0; i < npts; i++) {
        double xx[ndim];
        ind2sub(i, nn, indices);
        double jac = 1.0;
        for (int j = 0; j < ndim; j++) {
            // map to (0, pi)
            double x = (indices[j]+1) * h;
            // map to (-oo, oo)
            xx[j] = transform_cc(x);
            jac *= jacobian_cc(x);
        }
        total += jac*psi_fn(xx);
    }

    return total*std::pow(h, ndim);
}

int main()
{

    // Number of grid points in each dimension
    int n = 12;

#if 1
    // Use a single value of n
    uint64_t npts = get_npts(ndim, n);
    std::cout << "npts = " << npts << std::endl;
    double start = omp_get_wtime();
    double val = trapn_cc(n);
    double end = omp_get_wtime();
    std::cout << "val = " << std::setprecision(16) << val << std::endl;
    double elapsed = end-start;
    std::cout << " time = " << elapsed << "  eval rate = " << npts/elapsed << std::endl;
#endif


    // Scan different values of n
#if 0
    std::cout << "n   npts   val     elapsed time (s)    rate (evals/s) " << std::endl;
    for (int n = 10; n < 24; n++) {
      uint64_t npts = get_npts(ndim, n);
      double start = omp_get_wtime();
      double val = trapn_cc(n);
      double end = omp_get_wtime();
      double elapsed = end-start;
#ifdef USE_FMT_LIBRARY
      fmt::print("{0:<5} {1:<8} {2:12.8e} {3:7.4f} {4:6.3e}\n",n,npts,val,elapsed,npts/elapsed);
#else
      std::cout << n << " "
                << npts <<  " "
                << val << " "
                << elapsed << " "
                << npts/elapsed << std::endl;
#endif
    }
#endif


}
