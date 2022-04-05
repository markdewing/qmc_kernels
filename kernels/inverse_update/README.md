# Determinants, matrix inversion, and updates

The [Slater determinant](https://en.wikipedia.org/wiki/Slater_determinant) is an essential part of the wavefunction that enforces electron anti-symmetry.
For the kinetic part of the local energy, the derivatives are also needed.

The determinant can be computed from the LU decomposition, which is a also a step towards computing the inverse.

Through [Sherman-Morrison](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
and [Woodbury](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) matrix formulas,
the inverse matrix can be updated through single and multiple particle updates at less cost than a full inversion.
