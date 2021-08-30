
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

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
    return std::pow((n-1), ndim);
}

double transform_cc(double t)
{
    double L = 2.0;
    return L/std::tan(t);
}

double jacobian_cc(double t)
{
    double L = 2.0;
    double s = std::sin(t);
    return L/(s*s);
}

double trapn_cc(int n)
{
    double h = pi/n;

    double total = 0.0;

    int npts = std::pow((n-1), ndim);
    int indices[ndim];
    int nnm1[ndim];
    for (int i = 0; i < ndim; i++) {
        nnm1[i] = n-1;
    }


#pragma omp parallel for reduction(+:total)
    for (int i = 0; i < npts; i++) {
        double xx[ndim];
        ind2sub(i, nnm1, indices);
        double jac = 1.0;
        for (int j = 0; j < ndim; j++) {
            double x = (indices[j]+1) * h;
            xx[j] = transform_cc(x);
            jac *= jacobian_cc(x);
        }
        total += jac*psi_fn(xx);
    }

    return total*std::pow(h, ndim);
}

int main()
{

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


    // Scan different values of n
#if 0
    std::cout << "n   npts   val     elapsed time (s)    rate (evals/s) " << std::endl;
    for (int n = 10; n < 24; n++) {
      uint64_t npts = get_npts(ndim, n);
      double start = omp_get_wtime();
      double val = trapn_cc(n);
      double end = omp_get_wtime();
      double elapsed = end-start;
      std::cout << n << " "
                << npts <<  " "
                << val << " "
                << elapsed << " "
                << npts/elapsed << std::endl;
    }

#endif

}
