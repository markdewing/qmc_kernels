
// Vector add example in C++

#include <chrono>
#include <iostream>
#include <vector>

template <typename T>
void vector_add(const std::vector<T> &a, const std::vector<T> &b,
                std::vector<T> &c) {
  for (int i = 0; i < c.size(); i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {

  using RealType = double;
  const uint64_t N = 10 * 1000*1000ULL;
  std::vector<RealType> a(N);
  std::vector<RealType> b(N);
  std::vector<RealType> c(N);

  std::cout << "Array size = " << N * sizeof(RealType) * 1.0 / std::giga::num << " GB" << std::endl;

  std::fill(a.begin(), a.end(), 1.0);

  for (int i = 0; i < N; i++) {
    b[i] = 1.0 * i;
  }

  auto start = std::chrono::high_resolution_clock::now();

  vector_add(a, b, c);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::chrono::duration<double, std::nano> elapsed_ns = end - start;
  std::cout << "Elapsed time = " << elapsed_ns.count() << " ns" << std::endl;
  std::cout << "Elapsed time = " << elapsed_ns.count() / N << " ns per element"
            << std::endl;

  uint64_t read_bytes = 2 * N * sizeof(RealType);
  uint64_t write_bytes = N * sizeof(RealType);
  uint64_t rw_bytes = read_bytes + write_bytes;

  double bw = rw_bytes * 1.0 / elapsed.count();

  std::cout << " BW (GB/s) = " << bw / std::giga::num << std::endl;

  return 0;
}
