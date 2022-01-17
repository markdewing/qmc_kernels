
g++ \
-I /usr/include/hdf5/serial \
solve_linear_matrices.cpp \
-lhdf5_serial \
-lmkl_intel_lp64 -lmkl_gnu_thread -lgomp -lmkl_core
# With f77 LAPACK/BLAS
#-llapack -lblas \
