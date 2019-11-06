#!/bin/sh

# Using the Intel LLVM Sycl compiler from the sycl branch of https://github.com/intel/llvm
clang++ -fsycl  vector_add.cpp  -lOpenCL

