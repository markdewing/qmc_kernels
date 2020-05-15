
#include <iostream>
#include <vector>
#include <Kokkos_Core.hpp>

void run_vector_add()
{
  using RealType = float;

  const int N = 1024*1024*40;

  Kokkos::View<RealType*> a("a",N);
  Kokkos::View<RealType*> b("b",N);
  Kokkos::View<RealType*> c("c",N);

  Kokkos::View<RealType*>::HostMirror h_a = Kokkos::create_mirror_view(a);
  Kokkos::View<RealType*>::HostMirror h_b = Kokkos::create_mirror_view(b);
  Kokkos::View<RealType*>::HostMirror h_c = Kokkos::create_mirror_view(c);

  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0;
    h_b[i] = 1.0*i;
  }

  Kokkos::deep_copy(a, h_a);
  Kokkos::deep_copy(b, h_b);


  Kokkos::Timer timer;

  Kokkos::parallel_for("vector_add",N, KOKKOS_LAMBDA(int i) {
      c[i] = a[i] + b[i];
  });

  double elapsed = timer.seconds();
  Kokkos::deep_copy(h_c, c);

  std::cout << "c[0] = " << h_c[0] << std::endl;
  std::cout << "c[N-1] = " << h_c[N-1] << std::endl;


  std::cout << "kernel time = " << elapsed << std::endl;

  unsigned int all_bytes =  3*N*sizeof(RealType);
  std::cout <<"BW(GB/s) = " << all_bytes*1e-9/elapsed << std::endl;
} 

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);

  run_vector_add();

  Kokkos::finalize();

  return 0;
}
