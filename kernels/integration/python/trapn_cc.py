
import math
import time
import numpy as np

# Integrand
import psi


# Quadrature is Clenshaw-Curtis - trapezoidal rule from -oo to oo with n divisions
# https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature

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

def get_npts():
    return (n-1)**ndim

# Assume the integrand goes to zero faster than 1/x^2, so the endpoints evaluate to zero.

def trapn_cc(ndim, n, fn):
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
    for i in range(npts):
        idx = ind2sub(i, nnm1, idx) + 1
        x[:] = idx*h
        xt[:] = transform_cc(x)
        jac = np.prod(jacobian_cc(x))
        total += jac*fn(xt)

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
    n = 6

    npts = get_npts()
    print("npts: ", npts)
    start = time.perf_counter()
    val = trapn_cc(ndim, n, psi_fn)
    end = time.perf_counter()
    elapsed = end - start
    print('Integrated value: ',val)
    print('Elapsed time (s): ',elapsed,' rate (func evals/s): ',npts/elapsed)

