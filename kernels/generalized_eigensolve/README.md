

The linear method for optimizing parameters in QMCPACK yields a generalized eigenvalue problem to solve.

This solve becomes the bottleneck when scaling up in the number of parameters.




The entries in the matrices can be considered as the infinite sample value with noise.

The overlap matrix is symmetric.
The Hamiltonian matrix would be symmetric in the infinite sample limit.
The finite sample version is asymmetric.
Symmetrizing the matrix produces worse results because the asymmetric version is an unbiased estimator.

Is there be any algorithmic advantage from knowing the Hamiltonian is an almost-symmetric matrix?



Sample versions of the matrix can be obtained from QMCPACK using the "output_matrices_hdf" parameter.
The `linear_matrices.h5` file contains the overlap and Hamiltonian matrices, and other paramters necessary for recreating the QMCPACK solve.
It also contains the output values so other implementations can be checked against QMCPACK.


* `qmcpack_linear_method_solver.py` - Python implementation of the solver

* `solve_linear_matrices.cpp` - C++ implementation of the solver
