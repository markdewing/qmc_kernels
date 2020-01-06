
// Vector add example in C

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define RealType double
#define N 1000

double cpu_clock()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double) tv.tv_usec * 1e-6 + (double)tv.tv_sec;
}

void vector_add(RealType *a, RealType *b, RealType *c, int n)
{
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main()
{

  RealType *a = (RealType *)malloc(sizeof(RealType) * N);
  RealType *b = (RealType *)malloc(sizeof(RealType) * N);
  RealType *c = (RealType *)malloc(sizeof(RealType) * N);

  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 1.0*i;
  }

  double start = cpu_clock();

  vector_add(a, b, c, N);

  double end = cpu_clock();
  double elapsed = end - start;
  double elapsed_ns = elapsed * 1e9;

  printf(" Elapsed time = %g ns\n",elapsed_ns);
  printf(" Elapsed time = %g ns per element\n",elapsed_ns/N);

  return 0;
}
