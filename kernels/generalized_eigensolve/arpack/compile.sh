
ARPACK_ROOT=~/software/linalg/arnoldi/usr/
g++ \
-O2 \
-I /usr/include/hdf5/serial \
-I ${ARPACK_ROOT}/include \
solve_linear_matrices.cpp \
-lhdf5_serial \
-L${ARPACK_ROOT}/lib -larpack \
-llapack -lblas

#-DUSE_MKL -lmkl_intel_lp64 -lmkl_gnu_thread -lgomp -lmkl_core

# With f77 LAPACK/BLAS
#-llapack -lblas

# With MKL
#-DUSE_MKL -lmkl_intel_lp64 -lmkl_gnu_thread -lgomp -lmkl_core
# With serial MKL
#-DUSE_MKL -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
