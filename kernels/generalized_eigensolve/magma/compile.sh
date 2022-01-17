
g++ \
-I /usr/include/hdf5/serial \
-DUSE_MAGMA -I/home/mdewing/physics/codes/linalg/magma/usr/include  \
solve_linear_matrices.cpp \
-lhdf5_serial \
-L/home/mdewing/physics/codes/linalg/magma/usr/lib -lmagma \
-llapack -lblas


# With MAGMA
#-DUSE_MAGMA -I/home/mdewing/physics/codes/linalg/magma/usr/include  \
#-L/home/mdewing/physics/codes/linalg/magma/usr/lib -lmagma

# With f77 LAPACK/BLAS
#-llapack -lblas

# With MKL
#-DUSE_MKL -lmkl_intel_lp64 -lmkl_gnu_thread -lgomp -lmkl_core
