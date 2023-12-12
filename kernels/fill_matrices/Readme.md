

Extract fillOverlapHamiltonianMatrices function from QMCCostFunctionBatched

The loop structure and memory accesses are similar to a matrix multiply.

Use the "fill\_matrices" parameter to optimizer to output the inputs for this function.
(This parameter is only implemented in the "rot_det_speedup" branch, in this commit: https://github.com/markdewing/qmcpack/commit/dd92ad1454d8636704fcfc33fd974f933b19c57e )

The files will be have the suffix "param_deriv.h5".

The executable can optionally take the corresponding linear matrices file as a second parameter
and this will allow the function output to be verified.
