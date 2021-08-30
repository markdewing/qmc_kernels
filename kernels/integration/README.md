
Grid-based integration for a simple He wavefunction.

Also not really a QMCPACK kernel.

The vector_add kernel is trivial computationally and performance depends entirely on memory bandwidth.
This kernel should go to the opposite extreme and be very computational intense.

See https://github.com/QMCPACK/qmc_algorithms/blob/master/Variational/Variational_Helium.ipynb for more explanation.

For now the codes here are only the normalization integral (integral of psi squared).

To get the energy (integral on the numerator), we need to take derivatives of the wavefunction.  This can be done by
1. Compute by hand
2. Autodifferentiation
3. Symbol differentiation with code generation

The integration algorithms come from
https://github.com/markdewing/crispy-quadrature




