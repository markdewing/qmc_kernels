# Using Triton, a language focused on deep learning kernels.

# https://triton-lang.org/main/index.html
# https://github.com/triton-lang/triton

# This is pretty much the vector add example: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def test1():
    n = 1000
    a = torch.ones(n, device="cuda")
    b = torch.arange(n, device="cuda")

    output_torch = a + b
    output_triton = add(a, b)
    print("output", output_triton[0:10])
    print("diff ", torch.max(torch.abs(output_torch - output_triton)))


if __name__ == "__main__":
    test1()
