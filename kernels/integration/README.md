
Grid-based integration for a simple He wavefunction.

Not a QMCPACK kernel, but it solves the same problem in a different way.

The vector_add kernel is trivial computationally and performance depends entirely on memory bandwidth.
This kernel goes to the opposite extreme and is very computational intense.

See https://github.com/QMCPACK/qmc_algorithms/blob/master/Variational/Variational_Helium.ipynb for more explanation.

For now the codes here only compute the normalization integral (integral of psi squared).  Not that interesting by itself, but the code is simple.

To get the energy (integral on the numerator), we need to take derivatives of the wavefunction.  This can be done by
1. Compute by hand
2. Autodifferentiation
3. Symbolic differentiation with code generation

The integration algorithms come from
https://github.com/markdewing/crispy-quadrature




