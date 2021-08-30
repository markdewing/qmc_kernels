
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#ifdef USE_FMT_LIBRARY
#include <fmt/core.h>
#endif

// Can use the fmt library for formatting the printed output
// (C++20 compatible) https://github.com/fmtlib/fmt

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

// Problem-dependent convergence parameter
const double L = 0.5;

double transform_cc(double t)
{
    // Can cause performance and/or numerical stability problems
    //return L/std::tan(t);

    //return L*std::tan(pi/2 - t);

    // Use alternate expression for cotangent
    double sin_t;
    double cos_t;
    sincos(t, &sin_t, &cos_t);
    return L *cos_t/sin_t;
}

double jacobian_cc(double t)
{
    double s = std::sin(t);
    return L/(s*s);
}

// Transform infinite interval to -1,1

double transform_inf(double t)
{
  return t/(1.0 - t*t);
}

double jacobian_inf(double t)
{
  double denom = (1.0-t*t);
  return (1.0 + t*t)/(denom*denom);
}

// Trapezoidal rule over infinite interval.
double trapn_inf(int n)
{
    // Interval is (-1,1)
    double h = 2.0/(n+1);

    double total = 0.0;

    int npts = std::pow(n, ndim);
    int indices[ndim];
    int nn[ndim];
    for (int i = 0; i < ndim; i++) {
        nn[i] = n;
    }


#pragma omp parallel for reduction(+:total)
    for (int i = 0; i < npts; i++) {
        double xx[ndim];
        ind2sub(i, nn, indices);
        double jac = 1.0;
        for (int j = 0; j < ndim; j++) {
            // map to indices to interval (-1,1)
            double x = (indices[j]+1) * h - 1.0;
            // map to (-oo,oo)
            xx[j] = transform_inf(x);
            jac *= jacobian_inf(x);
        }
        double fn_val = psi_fn(xx);
        total += jac*fn_val;
    }

    return total*std::pow(h, ndim);
}

// Clenshaw-Curtis quadrature over an infinite integral
double trapn_cc(int n)
{
    double h = pi/n;

    double total = 0.0;

    int npts = std::pow(n, ndim);
    int indices[ndim];
    int nn[ndim];
    for (int i = 0; i < ndim; i++) {
        nn[i] = n;
    }


#pragma omp parallel for reduction(+:total)
    for (int i = 0; i < npts; i++) {
        double xx[ndim];
        ind2sub(i, nn, indices);
        double jac = 1.0;
        for (int j = 0; j < ndim; j++) {
            // map to (0,pi)
            double x = (indices[j]+1) * h;
            // map to (-oo,oo)
            xx[j] = transform_cc(x);
            jac *= jacobian_cc(x);
        }
        double fn_val = psi_fn(xx);
        total += jac*fn_val;
    }

    return total*std::pow(h, ndim);
}

int main()
{

#if 1
    // Number of points in each dimension
    int n = 12;

    uint64_t npts = get_npts(ndim, n);

    // Compute for a single value of n
    std::cout << "n = " << n << " npts = " << npts << std::endl;
    double start = omp_get_wtime();
    double val = trapn_cc(n);
    double end = omp_get_wtime();
    std::cout << "val = " << std::setprecision(16) << val << std::endl;
    double elapsed = end-start;
    std::cout << " time = " << elapsed << "  eval rate = " << npts/elapsed << std::endl;
#endif


    // Number of intevals in each dimension

    // - number of grid points = n + 1
    // - number of interior grid points = n -1
    // spacing (h) determined from 1/(number of grid points)
    // Scan different values of n
#if 0
#ifdef USE_FMT_LIBRARY
    fmt::print("{0:<5}  {1:<10} {2:10} val(inf)   elapsed time (s)    rate (evals/s) \n","n","npts","h");
#else
    std::cout << "n   npts   h val(inf)   elapsed time (s)    rate (evals/s)  val(cc) elapsed_time(s) rate(eval/s) " << std::endl;
#endif
    for (int n = 3; n < 24; n++) {
      uint64_t npts = get_npts(ndim, n);
      double h = 2.0/n;

      double start = omp_get_wtime();
      double val_inf = trapn_inf(n);
      double end = omp_get_wtime();
      double elapsed = end-start;

      double start_cc = omp_get_wtime();
      double val_cc = trapn_cc(n);
      double end_cc = omp_get_wtime();
      double elapsed_cc = end_cc-start_cc;

#ifdef USE_FMT_LIBRARY
      fmt::print("{0:<5} {1:<8} {2:7.5f} {3:12.8e} {4:7.4f} {5:6.3e} {6:12.10e} {7:7.4f} {8:7.4g}\n",n,npts,h,val_inf,elapsed,npts/elapsed,val_cc,elapsed_cc,npts/elapsed_cc);
#else
      std::cout << n << " "
                << npts <<  " "
                << h <<  " "
                << val_inf << " "
                << elapsed << " "
                << npts/elapsed << " "
                << val_cc << " "
                << elapsed_cc << " "
                << npts/elapsed_cc << std::endl;
#endif
    }

#endif

}
