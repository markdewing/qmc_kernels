
#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <vector>

using namespace cl;

int main()
{
  using RealType = double;
  const int N = 10000;
  std::vector<RealType> a(N);
  std::vector<RealType> b(N);
  std::vector<RealType> c(N);

  std::fill(a.begin(), a.end(), 1.0);

  for (int i = 0; i < N; i++) {
      b[i] = 1.0*i;
  }

  std::vector<cl::sycl::platform> platforms = cl::sycl::platform::get_platforms();
  for (const auto &plat : platforms)
  {
      std::cout << "Platform : " << plat.get_info<sycl::info::platform::name>() << std::endl;
      std::cout << "Vendor : " << plat.get_info<sycl::info::platform::vendor>() << std::endl;
      std::vector<sycl::device> devices = plat.get_devices();
      for (const auto& dev : devices) {
          std::cout << " Device : " << dev.get_info<sycl::info::device::name>() << std::endl;
      }
  }

  sycl::default_selector selector;
  //sycl::gpu_selector selector;
  //sycl::cpu_selector selector;
  {
    //sycl::queue myQueue(selector);
    sycl::queue myQueue(platforms[4].get_devices()[0]);

    std::cout << "Running on " << myQueue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    sycl::buffer<RealType> buffer_A{std::begin(a), std::end(a)};
    sycl::buffer<RealType> buffer_B{std::begin(b), std::end(b)};
    sycl::buffer<RealType> buffer_C{std::begin(c), std::end(c)};

    myQueue.submit([&](sycl::handler &h){
            auto acc_A = buffer_A.get_access<sycl::access::mode::read>(h);
            auto acc_B = buffer_B.get_access<sycl::access::mode::read>(h);
            auto acc_C = buffer_C.get_access<sycl::access::mode::write>(h);

#if 1
            h.parallel_for<class vector_add>(sycl::range<1> {N},
                 [=](sycl::id<1> i)
                    { acc_C[i] = acc_A[i] + acc_B[i];}
            );
#endif

#if 0
            // As a task
            h.single_task<class seq_vector_add>([=]() {
              for (size_t i = 0; i < N; i++) {
                acc_C[i] = acc_A[i] + acc_B[i];
              }
            });
#endif

        });

   auto host_C_acc = buffer_C.get_access<sycl::access::mode::read>();
   std::cout << "host_acc[0] = " << host_C_acc[0] << std::endl;
   std::cout << "host_acc[N-1] = " << host_C_acc[N-1] << std::endl;
  }
  std::cout << "c[0] = " << c[0] << std::endl;
  std::cout << "c[N-1] = " << c[N-1] << std::endl;
  return 0;
}
