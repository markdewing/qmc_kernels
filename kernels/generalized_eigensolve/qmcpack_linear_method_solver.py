
# Compare generalized eigenvalue problem from QMCPACK
# Comes from using linear method.

import numpy as np
import scipy.linalg
import h5py

f = h5py.File("linear_matrices.h5","r")


# Load matrices
ovlp = np.array(f['overlap'])
ham = np.array(f['Hamiltonian'])

# Get shifts
shift_i = f['bestShift_i'][()]
shift_s = f['bestShift_s'][()]

#shift_i = 1.0e-2
#shift_s = 1.0

print('shift_i = ',shift_i)
print('shift_s = ',shift_s)

d = ovlp.shape[0]
print('dim = ',d)

#
# Apply shifts
#

ham2 = ham + shift_i * np.identity(d)
ham3 = ham2 + shift_s*ovlp
# shift does not apply to the first row/column
ham3[:,0] = ham[:,0]
ham3[0,:] = ham[0,:]

for i in range(d):
    if ovlp[i,i] < 1e-5:
        print("small ovlp",i,ovlp[i,i])




# Solve generalized eigenvalue problem directly
def compute_eigen_generalized(ham, ovlp):
    out = scipy.linalg.eig(ham,ovlp)
    evals = out[0]
    r_evec = out[1]
    return evals, r_evec

# Convert generalized eigenvalue problem to regular eigenvalue problem
#  via inversion and multiplication of overlap matrix
def compute_eigen_with_inverse(ham, ovlp):
    r2 = np.dot(np.linalg.inv(ovlp), ham)
    r2 = r2.T
    print('prod(0,0) = ',r2[0,0],ham[0,0])
    #print('product')
    #print(r2.T)
    #for i in range(d):
    #    for j in range(d):
    #        print('prod ',i,j,r2[i,j])

    out = scipy.linalg.eig(r2)
    evals = np.real(out[0])
    r_evec = np.real(out[1])

    return evals, r_evec


# Use LAPACK dgeev instead of scipy.linalg.eig
def compute_eigen_with_dgeev(ham, ovlp):
    r2 = np.dot(np.linalg.inv(ovlp), ham)
    #r2 = r2.T
    #print('product')
    #print(r2.T)
    #for i in range(d):
    #    for j in range(d):
    #        print('prod ',i,j,r2[i,j])

    out = scipy.linalg.lapack.dgeev(r2)
    evals = out[0]
    r_evec = out[3]

    return evals, r_evec


#
# Get eigenvalues, eigenvectors
#

#evals, evec =  compute_eigen_generalized(ham3, ovlp)
evals, evec =  compute_eigen_with_inverse(ham3, ovlp)
#evals, evec =  compute_eigen_with_dgeev(ham3, ovlp)
    

#
# Find eigenvalue closest to H(0,0)
#
zerozero = ham[0,0]

mapped_evals = np.zeros(d)
for i in range(d):
    ev = evals[i]
    if ev < zerozero and ev > (zerozero - 1e2):
        mapped_evals[i] = (ev - zerozero + 2.0)*(ev - zerozero + 2.0)
    else:
        mapped_evals[i] = 1e6


#idx = np.argmin(evals)
idx = np.argmin(mapped_evals)
sorted_idx = np.argsort(mapped_evals)
for i in range(min(d,30)):
    idx1 = sorted_idx[i]
    print(i,idx1,mapped_evals[idx1],evals[idx1])


print('minimum eigenvalue ',evals[idx],'at idx',idx)
#print(evals)

#
# Compare eigenvalue with one from QMCPACK
#

lowest_eval = f['lowest_eigenvalue'][()]
print("from qmcpack, lowest eigenval = ",lowest_eval, " diff = ",lowest_eval - np.real(evals[idx]))

qmcpack_scaled_evec = np.array(f['scaled_eigenvector'])


#
# Compare eigenvector with one from QMCPACK
#

#print('raw right evec')
#print(r_evec[:,idx])

print('Scaled eigenvector')
scaled_evec = evec[:,idx]/evec[0,idx]
#print(scaled_evec)
#print('diff from QMCPACK = ',np.abs(qmcpack_scaled_evec - scaled_evec))
print('norm of diff from QMCPACK= ',np.linalg.norm(np.abs(qmcpack_scaled_evec - scaled_evec)))

