from __future__ import print_function
import numpy as np
import numba
from timeit import default_timer

@numba.jit(nopython=True)
def vector_add(a,b,c,n):
    for i in range(n):
        c[i] = a[i] + b[i] 

def vector_add_bare(a,b,c,n):
    for i in range(n):
        c[i] = a[i] + b[i] 


def test_vector_add():
    n = 10000
    a = np.array([i for i in range(n)], dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    #a = [i for i in range(n)]
    #b =[1.0]*n
    #c = [0.0]*n

    n_iter = 10
    start_bare = default_timer()
    for i in range(n_iter):
        vector_add_bare(a,b,c,n)
    end_bare = default_timer()

    start = default_timer()
    for i in range(n_iter):
        vector_add(a,b,c,n)
    end = default_timer()

    print('c[0] = ',c[0])
    print('c[-1] = ',c[-1])
    print('python time = ',(end-start)/n_iter)
    print('jit    time = ',(end_bare-start_bare)/n_iter)


if __name__ == '__main__':
    test_vector_add()
