
# Different variants to update an inverse matrix

import numpy as np
import scipy.linalg.blas

def run_inverse_test():
  M = 4  # size of matrix
  #A = np.random.rand(M, M)
  A = np.array([
  [3.9080e-14, 9.8539e-04, 4.1631e-02, 1.7664e-01],
  [3.6460e-01, 9.1331e-02, 9.2298e-02, 4.8722e-01],
  [5.2675e-01, 4.5443e-01, 2.3318e-01, 8.3129e-01],
  [9.3173e-01, 5.6806e-01, 5.5609e-01, 5.0832e-02],
  ])

  Ainv = np.linalg.inv(A)

  # Choose row index to update
  p = 0

  # new row delta (note: not new row value)
  #vk = np.random.rand(M)
  vk = np.array([7.6705e-01 , 1.8915e-02,  2.5236e-01,  2.9820e-01])

  # Get the indexed column of the inverse matrix
  uk = Ainv[:,p]

  # Updated matrix.  Not needed for the update algorithms, but used for
  #   comparison in this script
  A_new = np.copy(A)
  print('A slice',A_new[p,:])
  A_new[p,:] += vk
  # new row value (not delta)
  b = np.copy(A_new[p,:])
  print('b',b)

  R = test_determinant_ratio(A, p, vk, uk, A_new)

  AinvSM = update_inverse_sm(Ainv, p, vk, uk, R)
  test_inverse(np.linalg.inv(A_new), AinvSM, 'Sherman-Morrison formula')

  Ainv_loop = update_inverse_loop(Ainv, p, b, R)
  test_inverse(np.linalg.inv(A_new), Ainv_loop, 'explicit loop')

  Ainv_blas = update_inverse_blas(np.copy(Ainv), p, b, R)
  test_inverse(np.linalg.inv(A_new), Ainv_blas, 'BLAS')


def test_determinant_ratio(A, p, vk, uk, A_copy):
  # determinant ratio using matrix determinant lemma
  # https://en.wikipedia.org/wiki/Matrix_determinant_lemma
  R = 1 + np.dot(vk, uk)

  # Direct computation of determinant ratio
  R_direct = np.linalg.det(A_copy)/np.linalg.det(A)

  if not np.isclose(R, R_direct):
    print('Error in ratio')

  print('Direct determinant ratio = ',R_direct)
  print('Determinant ratio using formula = ',R)

  return R

# Use the Sherman-Morrison formula to update inverse
# https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
def update_inverse_sm(Ainv, p, vk, uk, R):
  Ainv2 = Ainv - np.dot(np.outer(uk, vk), Ainv)/R
  return Ainv2


# Use explicit loops
def update_inverse_loop(Ainv, p, b, R):
  Ainv2 = np.zeros(Ainv.shape)

  M = Ainv.shape[0]

  # scale column p
  for j in range(M):
    Ainv2[j,p] = Ainv[j,p]/R

  for i in range(M):
    for j in range(M):
      if j != p:
        dp = 0.0
        for l in range(M):
          dp += Ainv[l,j]*b[l]
        Ainv2[i,j] = Ainv[i,j] - Ainv[i,p]*dp/R

  return Ainv2


# Using BLAS
# in-place update modifies Ainv
#   At least if overwrite_a is set in the dger call (but that flag doens't seem to work)
def update_inverse_blas(Ainv, p, b, R):
  y = np.zeros(Ainv.shape[0])
  print('b = ',b)
  scipy.linalg.blas.dgemv(alpha=1.0/R, a=Ainv, x=b, y=y, trans=1, overwrite_y=1)
  print('y = ',y)

  y[p] = 1 - 1.0/R

  rcopy = np.copy(Ainv[:,p])
  print('rcopy = ',rcopy)

  Ainv2 = scipy.linalg.blas.dger(alpha=-1.0, x=rcopy, y=y, a=Ainv)

  return Ainv2



def test_inverse(Ainv2, Ainv, desc):
  if not np.allclose(Ainv, Ainv2):
    print('Inverses are different from ',desc)
    print('Direct inverse')
    print(Ainv2)
    print('Inverse via updates')
    print(Ainv)
  else:
    print('Inverse passes for ',desc)


if __name__ == '__main__':
  run_inverse_test()
