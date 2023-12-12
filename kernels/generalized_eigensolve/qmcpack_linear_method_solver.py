# Compare generalized eigenvalue problem from QMCPACK
# Comes from using linear method.

import numpy as np
import scipy.linalg
import h5py
import sys
import argparse


def load_linear_matrices(fname):
    print("Loading from  ", fname)

    f = h5py.File(fname, "r")

    # Load matrices
    ovlp = np.array(f["overlap"])
    ham = np.array(f["Hamiltonian"])

    # Get shifts
    shift_i = f["bestShift_i"][()]
    shift_s = f["bestShift_s"][()]

    print("shift_i = ", shift_i)
    print("shift_s = ", shift_s)

    d = ovlp.shape[0]
    print("dim = ", d)

    lowest_eval = f["lowest_eigenvalue"][()]
    qmcpack_scaled_evec = np.array(f["scaled_eigenvector"])

    #
    # Apply shifts
    #

    ham2 = ham + shift_i * np.identity(d)
    ham3 = ham2 + shift_s * ovlp
    # shift does not apply to the first row/column
    ham3[:, 0] = ham[:, 0]
    ham3[0, :] = ham[0, :]

    for i in range(d):
        if ovlp[i, i] == 0.0:
            ovlp[i, i] = shift_i * shift_s
        if ovlp[i, i] < 1e-5:
            print("small ovlp", i, ovlp[i, i])

    return ovlp, ham3, (lowest_eval, qmcpack_scaled_evec)


# Solve generalized eigenvalue problem directly
def compute_eigen_generalized(ham, ovlp):
    out = scipy.linalg.eig(ham, ovlp)
    evals = out[0]
    r_evec = out[1]
    return evals, r_evec


# Convert generalized eigenvalue problem to regular eigenvalue problem
#  via inversion and multiplication of overlap matrix
def compute_eigen_with_inverse(ham, ovlp):
    r2 = np.dot(np.linalg.inv(ovlp), ham)
    print("prod(0,0) = ", r2[0, 0], ham[0, 0])
    # print('product')
    # print(r2.T)
    # for i in range(d):
    #    for j in range(d):
    #        print('prod ',i,j,r2[i,j])

    out = scipy.linalg.eig(r2)
    evals = np.real(out[0])
    r_evec = np.real(out[1])

    return evals, r_evec


def compute_eigen_with_arpack(ham, ovlp):
    #r2 = np.dot(np.linalg.inv(ovlp), ham)
    #r2 = r2.T
    #print("prod(0,0) = ", r2[0, 0], ham[0, 0])
    zerozero = ham[0, 0]
    # print('product')
    # print(r2.T)
    # for i in range(d):
    #    for j in range(d):
    #        print('prod ',i,j,r2[i,j])

    # out = scipy.linalg.eig(r2)
    # out = scipy.sparse.linalg.eigs(r2,k=100,sigma=zerozero-100.0,which='LM',return_eigenvectors=True)
    out = scipy.sparse.linalg.eigs(
        ham, k=10, M=ovlp, sigma=zerozero - 100.0, which="LR", return_eigenvectors=True
    )
    # evals = np.real(out[0])
    # r_evec = np.real(out[1])
    w = out[0]
    r_evec = out[1]
    print("eigenvalues", w)

    return w, r_evec


# Use LAPACK dgeev instead of scipy.linalg.eig
def compute_eigen_with_dgeev(ham, ovlp):
    r2 = np.dot(np.linalg.inv(ovlp), ham)
    # r2 = r2.T
    # print('product')
    # print(r2.T)
    # for i in range(d):
    #    for j in range(d):
    #        print('prod ',i,j,r2[i,j])

    out = scipy.linalg.lapack.dgeev(r2)
    evals = out[0]
    r_evec = out[3]

    return evals, r_evec


# Get lowest eigenvalue and corresponding eigenvector
def get_lowest_eigenvector(ham, ovlp, method):
    print(f"Using method: {method}")
    need_mapping = False
    if method == "general":
        evals, evec = compute_eigen_generalized(ham, ovlp)
        need_mapping = True
    elif method == "inverse":
        evals, evec = compute_eigen_with_inverse(ham, ovlp)
        need_mapping = True
    elif method == "dgeev":
        evals, evec = compute_eigen_with_dgeev(ham, ovlp)
        need_mapping = True
    elif method == "arpack":
        evals, evec = compute_eigen_with_arpack(ham, ovlp)
        lowest_eval = evals[0]
        lowest_evec = evec[:, 0]
        print("lowest eigenvalue = ", lowest_eval)
    else:
        raise RuntimeError(f"Method not found: {method}")

    # When solving the full eigenvalue problem, need to find
    # the correct eigenvalue
    if need_mapping:

        # Find eigenvalue closest to H(0,0)
        zerozero = ham[0, 0]

        print("evec shape", evec.shape)
        d = evec.shape[1]
        mapped_evals = np.zeros(d)
        for i in range(d):
            ev = evals[i].real
            if ev < zerozero and ev > (zerozero - 1e2):
                mapped_evals[i] = (ev - zerozero + 2.0) * (ev - zerozero + 2.0)
            else:
                mapped_evals[i] = 1e6

        print("Original eigenvalues")
        for i in range(min(d, 10)):
            print(i, evals[i])

        # idx = np.argmin(evals)
        idx = np.argmin(mapped_evals)
        sorted_idx = np.argsort(mapped_evals)
        print("Mapped eigenvalues")
        for i in range(min(d, 10)):
            idx1 = sorted_idx[i]
            print(i, idx1, mapped_evals[idx1], evals[idx1])

        print("minimum eigenvalue ", evals[idx], "at idx", idx)
        # print(evals)

        lowest_eval = evals[idx]
        lowest_evec = evec[:, idx]

    return lowest_eval, lowest_evec


def compute_eigenvector(ovlp, ham, evec_info, method):
    lowest_eval, evec = get_lowest_eigenvector(ham, ovlp, method)

    qmcpack_lowest_eval = evec_info[0]
    qmcpack_scaled_evec = evec_info[1]

    #
    # Compare eigenvalue with one from QMCPACK
    #

    print(
        "lowest eigenvalue from qmcpack = ",
        qmcpack_lowest_eval,
        " diff = ",
        lowest_eval - qmcpack_lowest_eval,
    )

    #
    # Compare eigenvector with one from QMCPACK
    #

    # print('raw right evec')
    # print(r_evec[:,idx])

    print("Scaled eigenvector")
    scaled_evec = evec / evec[0]
    # print(scaled_evec)
    # print('diff from QMCPACK = ',np.abs(qmcpack_scaled_evec - scaled_evec))

    print(
        "norm of diff from QMCPACK= ",
        np.linalg.norm(np.abs(qmcpack_scaled_evec - scaled_evec)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve generalized eigenvalue problem from QMCPACK"
    )
    parser.add_argument("input_file", help="Input HDF file: *.linear_matrices.h5")
    parser.add_argument(
        "-m", "--method", help="Method for eigevalue computation (general, inverse, dgeev, arpack)", default="general"
    )

    args = parser.parse_args()
    fname_in = args.input_file

    method = args.method

    ovlp, ham, evec_info = load_linear_matrices(fname_in)
    compute_eigenvector(ovlp, ham, evec_info, method)
