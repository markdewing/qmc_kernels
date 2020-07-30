#!/bin/sh

clang++ -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda  --cuda-path=/usr/local/cuda-10.1  vector_add.cpp
# add -v to see all the toolchain steps
# add -save-temps=cwd to keep all the intermediate files in the current directory
