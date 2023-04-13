
// Vector add example using OpenMP target offload

#include <chrono>
#include <iostream>
#include <vector>

void vector_add(float *a, float *b, float *c, int n) {
#pragma omp target teams distribute parallel for map(to : a [0:n], b [0:n]) map(from : c[0:n])
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {

  using RealType = float;
  const int N = 1024 * 1024 * 40;
  std::vector<RealType> a(N);
  std::vector<RealType> b(N);
  std::vector<RealType> c(N);

  std::fill(a.begin(), a.end(), 1.0);

  for (int i = 0; i < N; i++) {
    b[i] = 1.0 * i;
  }

  auto start = std::chrono::high_resolution_clock::now();

  vector_add(a.data(), b.data(), c.data(), N);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::chrono::duration<double, std::nano> elapsed_ns = end - start;
  std::cout << "Elapsed time = " << elapsed_ns.count() << " ns" << std::endl;
  std::cout << "Elapsed time = " << elapsed_ns.count() / N << " ns per element"
            << std::endl;

  int read_bytes = 2 * N * sizeof(RealType);
  int write_bytes = N * sizeof(RealType);
  int rw_bytes = read_bytes + write_bytes;

  double bw = rw_bytes * 1.0 / elapsed.count();

  std::cout << " BW (GB/s) = " << bw / std::giga::num << std::endl;

  return 0;
}
