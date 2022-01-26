mpiicpc \
-O2 -g \
-I /usr/include/hdf5/serial \
-D USE_MKL=1 \
-I/opt/intel/oneapi/mpi/latest/include/ \
-I/opt/intel/oneapi/mkl/latest/include/ \
solve_linear_matrices.cpp   \
-L/opt/intel/oneapi/mpi/latest/lib \
-L/opt/intel/oneapi/mkl/latest/lib/intel64 \
-lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lhdf5_serial


#-L/home/mdewing/physics/codes/linalg/scalapack/usr/lib -lscalapack -lgfortran -llapack -lblas
