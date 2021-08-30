
import math
import time
import numpy as np
import multiprocessing as mp

# Integrand
import psi

# Quadrature is Clenshaw-Curtis - trapezoidal rule from -oo to oo with n divisions
# https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature

# In N dimensions

# Coordinate transformation
@np.vectorize
def transform_cc(t):
    L = 2.0
    return L/np.tan(t)

# Jacobian of the transformation
@np.vectorize
def jacobian_cc(t):
    L = 2.0
    return L/np.sin(t)**2

# https://stackoverflow.com/questions/46782444/how-to-convert-a-linear-index-to-subscripts-with-support-for-negative-strides
def ind2sub(idx, shape, indices):
    for i in range(len(shape)):
        s = idx % shape[i]
        idx -= s
        idx //= shape[i]
        indices[i] = s
    return indices

# Assume the integrand goes to zero faster than 1/x^2, so the endpoints evaluate to zero.

def trapn_cc_proc(start, end, fn, nnm1, ndim, h, res_queue):

    idx = np.zeros(ndim, dtype=np.int64)
    x = np.zeros(ndim)
    xt = np.zeros(ndim)

    total = 0.0
    for i in range(start,end):
        idx = ind2sub(i, nnm1, idx) + 1
        x[:] = idx*h
        xt[:] = transform_cc(x)
        jac = np.prod(jacobian_cc(x))
        total += jac*fn(xt)

    res_queue.put(total)

def get_npts():
    return (n-1)**ndim

def trapn_cc_mp(ndim, n, fn, nproc):
    a = 0.0
    b = math.pi
    # Size of each interval
    h = (1.0/n)*(math.pi)
    hval = h**ndim

    total = 0.0
    nnm1 = [n-1]*ndim
    idx = np.zeros(ndim, dtype=np.int64)
    x = np.zeros(ndim)
    xt = np.zeros(ndim)

    npts = (n-1)**ndim

    pts_per_proc = npts//nproc

    res_queue = mp.Queue()
    all_p = list()

    start = 0
    for i in range(nproc):
        end = min(start + pts_per_proc, npts)
        p = mp.Process(target=trapn_cc_proc, args=(start, end, fn, nnm1, ndim, h, res_queue))
        all_p.append(p)
        p.start()

        start += pts_per_proc

    for i in range(nproc):
        all_p[i].join()


    total = 0.0
    for i in range(nproc):
        total += res_queue.get()


    #for i in range(npts):
    #    idx = ind2sub(i, nnm1, idx) + 1
    #    x[:] = idx*h
    #    xt[:] = transform_cc(x)
    #    jac = np.prod(jacobian_cc(x))
    #    total += jac*fn(xt)

    return total*hval


def psi_fn(x):
    r1 = x[0:3]
    r2 = x[3:6]
    B = 0.1

    p = psi.psi(r1, r2, B)
    return p*p


if __name__ == '__main__':
    # Fixed for this integrand (2 particles in 3D space)
    ndim = 6

    # Number of points in each dimension
    n = 12

    # Number of parallel instances
    nproc = mp.cpu_count()
    print('Number of parallel instances: ',nproc)

    do_single_n = True
    if do_single_n:
        npts = get_npts()
        print("Using multiprocessing for parallel evaluation of integrand")
        print("npts: ", npts)
        start = time.perf_counter()
        val = trapn_cc_mp(ndim, n, psi_fn, nproc)
        end = time.perf_counter()
        elapsed = end - start
        print('Integrated value: ',val)
        print('Elapsed time (s): ',elapsed,' rate (func evals/s): ',npts/elapsed)

    do_scan_n = False
    if do_scan_n:
        print('# n  npts  val  time (s)   rate (func eval/s)')
        for n in range(10,18):
            npts = get_npts()
            start = time.perf_counter()
            val = trapn_cc_mp(ndim, n, psi_fn, nproc)
            end = time.perf_counter()
            elapsed = end - start
            print(n,npts,val,elapsed,npts/elapsed)






