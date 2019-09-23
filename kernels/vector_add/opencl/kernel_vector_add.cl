
__kernel void vector_add(__global const float *a, __global const float *b, __global float *c, int n)
{
  int id = get_global_id(0);

  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
