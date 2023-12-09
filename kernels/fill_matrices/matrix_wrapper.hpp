
// Stupidly simple wrapper to do vector, matrix, and tensor indexing

#include <iostream>
#include <vector>

#define BOUNDS_CHECK

#ifdef BOUNDS_CHECK
void bounds_check(int i, int D1, int dim_idx) {
  if (i >= D1) {
    std::cout << "out of bounds " << i << " dim " << D1 << " on dimension "
              << dim_idx << std::endl;
  }
}
#else
void bounds_check(int i, int D1, int dim_idx) {}
#endif

template <class T> class Wrapper1D {
public:
  Wrapper1D(int N, T *data) : D1(N), data_(data) {}

  T *data() { return data_; }
  T *data_;

  int D1;
  T operator()(int i) const {
    bounds_check(i, D1, 0);
    return data_[i];
  }

  T &operator()(int i) {
    bounds_check(i, D1, 0);
    return data_[i];
  }
};

template <class T> class Wrapper2D {
public:
  Wrapper2D(int N, int M, T *data) : D1(N), D2(M), data_(data) {}

  T *data() { return data_; }
  T *data_;

  int D1;
  int D2;
  T operator()(int i, int j) const {
    bounds_check(i, D1, 0);
    bounds_check(j, D2, 1);
    // Row-major storage
    return data_[i * D2 + j];
    // Column-major storage
    // return data_[i + j*D1];
  }

  T &operator()(int i, int j) {
    bounds_check(i, D1, 0);
    bounds_check(j, D2, 1);
    // Row-major storage
    return data_[i * D2 + j];
    // Column-major storage
    // return data_[i + j*D1];
  }

  int rows() const { return D1; }
  int cols() const { return D2; }
};

template <class T> class Wrapper3D {
public:
  Wrapper3D(int N, int M, int K, T *data) : D1(N), D2(M), D3(K), data_(data) {}

  T *data() { return data_; }
  T *data_;

  int D1;
  int D2;
  int D3;
  T operator()(int i, int j, int k) const {
    bounds_check(i, D1, 0);
    bounds_check(j, D2, 1);
    bounds_check(k, D3, 2);
    // Row-major storage
    return data_[(i * D3 + j) * D2 + k];
    // Column-major storage
    // return data_[i + j*D1];
  }

  T &operator()(int i, int j, int k) {
    bounds_check(i, D1, 0);
    bounds_check(j, D2, 1);
    bounds_check(k, D3, 2);
    // Row-major storage
    return data_[(i * D3 + j) * D2 + k];
    // Column-major storage
    // return data_[i + j*D1];
  }
};
