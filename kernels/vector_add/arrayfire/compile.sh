#!/bin/sh

g++ \
-L /opt/arrayfire/lib64/  \
-I /opt/arrayfire/include/ \
vector_add.cpp -laf

