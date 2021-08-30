#!/bin/bash


# Using AMD AOMP compiler
aompcc -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx900 \
  offload_trapn_cc.cpp -o offload_a.out
