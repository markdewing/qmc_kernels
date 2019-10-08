

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

kernel_str = """
    __global__ void vector_add(const float *a, const float *b, float *c, int N)
    {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       if (idx < N) {
         c[idx] = a[idx] + b[idx];
       }
    }
"""


def test_vector_add():
    print("Max block dim x = ",cuda.Device(0).get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X))
    n = 1024*1024
    a = np.array([i for i in range(n)], dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    print("Vector size (MB) = ",a.nbytes/1024.0/1024.0)

    mod = SourceModule(kernel_str)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    vector_add_func = mod.get_function("vector_add")

    start = cuda.Event()
    end = cuda.Event()

    start.record()
    vector_add_func(a_gpu, b_gpu, c_gpu, np.int32(n), block=(1024,1,1),grid=(int(n/1024.0)+1,1))
    end.record()
    end.synchronize()
    kernel_ms = end.time_since(start)
    print("kernel ms = ",kernel_ms)

    bw_bytes = a.nbytes + b.nbytes + c.nbytes
    print("BW (GB/s) = ",bw_bytes*1e-6/kernel_ms)

    cuda.memcpy_dtoh(c, c_gpu)
    print("c[0] = ",c[0])
    print("c[-1] = ",c[-1])



if __name__ == '__main__':
    test_vector_add()
