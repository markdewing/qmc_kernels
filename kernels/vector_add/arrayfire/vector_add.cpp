
#include <arrayfire.h>
#include <iostream>

int main()
{
  af::setBackend(AF_BACKEND_CPU);
  //af::setBackend(AF_BACKEND_CUDA);
  //set AF_OPENCL_DEFAULT_DEVICE to choose an OpenCL backend
  //af::setBackend(AF_BACKEND_OPENCL);
  af::info();
  const int N = 10000;
  af::array a(N);
  af::array b(N);
  af::array c(N);

  for (int i = 0; i < N; i++) {
    a(i) = 1.0;
    b(i) = 1.0 * i;
  }

  af::timer t1 = af::timer::start();
  c = a + b;
  double elapsed = t1.stop();
  std::cout << " Elapsed time = " << elapsed*1e6 << " us " << std::endl;

  af_print(c(0));
  af_print(c(N-1));

  return 0;
}
