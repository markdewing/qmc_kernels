# Python implementation of filling the Hamiltonian and overlap matrices
#  for the linear method

# Equation numbers from Toulouse and Umrigar
#  https://arxiv.org/abs/physics/0701039
import numpy as np
import h5py
import sys
import argparse

# import numba

# Indices into SumValue
SUM_E_BARE = 0
SUM_ESQ_BARE = 1
SUM_ABSE_BARE = 2
SUM_E_WGT = 3
SUM_ESQ_WGT = 4
SUM_ABSE_WGT = 5
SUM_WGT = 6
SUM_WGTSQ = 7

# Indices into RecordsOnNode
LOGPSI_FIXED = 0
LOGPSI_FREE = 1
ENERGY_TOT = 2
ENERGY_FIXED = 3
ENERGY_NEW = 4
REWEIGHT = 5


def load_param_deriv(fname):
    print("Loading from ", fname)

    f = h5py.File(fname, "r")

    SumValue = np.array(f["SumValue"])
    records_on_node = np.array(f["RecordsOnNode"])
    w_beta = f["w_beta"][()]
    deriv_records = np.array(f["DerivRecords"])
    H_deriv_records = np.array(f["HDerivRecords"])

    print("DerivRecords shape", deriv_records.shape)
    print("w_beta", w_beta)

    return (
        deriv_records,
        H_deriv_records,
        records_on_node,
        SumValue,
        w_beta,
    )


# @numba.jit
def fill_matrices(deriv_records, H_deriv_records, records_on_node, SumValue, w_beta):
    nsamples, numParams = deriv_records.shape
    # numParams = 50

    b2 = w_beta
    if w_beta != 0.0:
        print("Only implemented for beta = 0.0")
        return

    curAvg_w = SumValue[SUM_E_WGT] / SumValue[SUM_WGT]

    D_avg = np.zeros(numParams)

    wgtinv = 1.0 / SumValue[SUM_WGT]

    for iw in range(nsamples):
        weight = records_on_node[iw, REWEIGHT] * wgtinv
        for pm in range(numParams):
            D_avg[pm] += deriv_records[iw, pm] * weight

    left = np.zeros((numParams + 1, numParams + 1))
    right = np.zeros((numParams + 1, numParams + 1))
    right[0, 0] = 1.0
    left[0, 0] = curAvg_w
    for iw in range(nsamples):
        weight = records_on_node[iw, REWEIGHT] * wgtinv
        eloc_new = records_on_node[iw, ENERGY_NEW]

        for pm in range(numParams):
            wfe = (
                H_deriv_records[iw, pm] + (deriv_records[iw, pm] - D_avg[pm]) * eloc_new
            ) * weight

            wfd = (deriv_records[iw, pm] - D_avg[pm]) * weight
            # Hamiltonian
            # eqn 54b ?
            left[0, pm + 1] += wfe
            # eqn 54c ?
            left[pm + 1, 0] += wfd * eloc_new

            for pm2 in range(numParams):
                # Hamiltonian (eqn 54d?)
                left[pm + 1, pm2 + 1] += wfd * (
                    H_deriv_records[iw, pm2]
                    + (deriv_records[iw, pm2] - D_avg[pm2]) * eloc_new
                )
                # Overlap (eqn 53c?)
                ovlij = wfd * (deriv_records[iw, pm2] - D_avg[pm2])
                right[pm + 1, pm2 + 1] += ovlij

    return right, left


# Loop over samples as the inner loop
# @numba.jit
def fill_matricesT(deriv_records, H_deriv_records, records_on_node, SumValue, w_beta):
    nsamples, numParams = deriv_records.shape
    # numParams = 50

    b2 = w_beta
    if w_beta != 0.0:
        print("Only implemented for beta = 0.0")
        return

    curAvg_w = SumValue[SUM_E_WGT] / SumValue[SUM_WGT]

    D_avg = np.zeros(numParams)

    wgtinv = 1.0 / SumValue[SUM_WGT]

    for iw in range(nsamples):
        weight = records_on_node[iw, REWEIGHT] * wgtinv
        for pm in range(numParams):
            D_avg[pm] += deriv_records[iw, pm] * weight
    # print('D_avg',D_avg)

    left = np.zeros((numParams + 1, numParams + 1))
    right = np.zeros((numParams + 1, numParams + 1))
    right[0, 0] = 1.0
    left[0, 0] = curAvg_w
    for pm in range(numParams):
        for iw in range(nsamples):
            weight = records_on_node[iw, REWEIGHT] * wgtinv
            eloc_new = records_on_node[iw, ENERGY_NEW]

            wfe = (
                H_deriv_records[iw, pm] + (deriv_records[iw, pm] - D_avg[pm]) * eloc_new
            ) * weight

            wfd = (deriv_records[iw, pm] - D_avg[pm]) * weight
            # Hamiltonian
            # eqn 54b ?
            left[0, pm + 1] += wfe
            # eqn 54c ?
            left[pm + 1, 0] += wfd * eloc_new

    for pm in range(numParams):
        for pm2 in range(numParams):
            right1 = 0.0
            left1 = 0.0
            for iw in range(nsamples):
                weight = records_on_node[iw, REWEIGHT] * wgtinv
                eloc_new = records_on_node[iw, ENERGY_NEW]
                wfd = (deriv_records[iw, pm] - D_avg[pm]) * weight
                # Hamiltonian (eqn 54d?)
                left1 += wfd * (
                    H_deriv_records[iw, pm2]
                    + (deriv_records[iw, pm2] - D_avg[pm2]) * eloc_new
                )
                # Overlap (eqn 53c?)
                right1 += wfd * (deriv_records[iw, pm2] - D_avg[pm2])

            right[pm + 1, pm2 + 1] = right1
            left[pm + 1, pm2 + 1] = left1

    return right, left


def load_reference(fname):
    print("Loading reference from ", fname)
    f = h5py.File(fname, "r")

    ovlp = np.array(f["overlap"])
    ham = np.array(f["Hamiltonian"])

    return ovlp, ham


def check_matrices(ovlp, ham, gold_ovlp, gold_ham):
    print("ovlp close:", np.allclose(ovlp, gold_ovlp))
    print("ham close:", np.allclose(ham, gold_ham))
    print("ovlp diff", np.linalg.norm(ovlp - gold_ovlp))
    print("ham diff", np.linalg.norm(ham - gold_ham))
    # print('gold ovlp')
    # print(gold_ovlp)
    # print('ovlp')
    # print(ovlp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct overlap and Hamiltonian matrices"
    )

    parser.add_argument("input_file", help="Input HDF file: *.param_deriv.h5")
    parser.add_argument(
        "-r", "--reference", help="reference HDF file: *.linear_matrices.h5"
    )

    args = parser.parse_args()

    param = load_param_deriv(args.input_file)
    if args.reference is not None:
        gold_ovlp, gold_ham = load_reference(args.reference)

    ovlp, ham = fill_matrices(*param)
    # ovlp, ham = fill_matricesT(*param)

    if args.reference is not None:
        check_matrices(ovlp, ham, gold_ovlp, gold_ham)
