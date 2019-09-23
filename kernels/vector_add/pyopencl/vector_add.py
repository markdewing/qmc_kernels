
from __future__ import print_function

import pyopencl as cl
import numpy as np
import timeit

kernel_str = """
__kernel void vector_add(__global const float *a, __global const float *b, __global float *c, int n)
{
  int id = get_global_id(0);

  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
"""


def run_vector_add():
    ctx = cl.create_some_context(interactive=False);

    queue =  cl.CommandQueue(ctx)

    prg = cl.Program(ctx, kernel_str).build()

    n = 1000
    host_a = np.array([i for i in range(n)], dtype=np.float32)
    host_b = np.ones(n, dtype=np.float32)

    device_a = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_a)
    device_b = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_b)

    device_c = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, host_a.nbytes)

    host_start = timeit.default_timer()

    prg.vector_add(queue, host_a.shape, None, device_a, device_b, device_c, np.int32(n))

    host_end = timeit.default_timer()

    host_c = np.empty_like(host_a)
    cl.enqueue_copy(queue, host_c, device_c)

    print("result[0] = ",host_c[0])
    print("result[-1] = ",host_c[-1])


    print("host time = ",host_end - host_start," s")
if __name__ == '__main__':
    run_vector_add()
