
import numpy as np

def mag(r):
    return np.sqrt(r[0]*r[0] + r[1]*r[1] +  r[2]*r[2])

def orb(R):
    r = mag(R)
    Z = 2.0
    norm = 0.5/np.sqrt(np.pi)
    y = norm*np.exp(-Z*r)
    return y

def jastrow(r12, B):
    A = 0.5
    return np.exp(A*r12/(1 + B*r12) - A/B)

def psi(r1, r2, B):
    o1 = orb(r1)
    o2 = orb(r2)
    r12 = r2 - r1
    r12_mag = mag(r12)
    j = jastrow(r12_mag, B)
    return o1*o2*j

