#!/bin/sh

clang++ -O2 --cuda-path=/usr/local/cuda-10.1 --cuda-gpu-arch=sm_35   vector_add.cu -L/usr/local/cuda-10.1/lib64 -lcudart_static -ldl -lrt -lpthread 

# Add -S to get PTX assembly view
