#!/bin/sh

nvcc -O2 vector_add.cu

# add -ptx to see the PTX output (in vector_add.ptx)
# add -v to see all the toolchain commands
# add -keep to save all the temporary files (placed in this directory)
