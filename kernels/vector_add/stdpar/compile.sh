
# NVidia NVHPC SDK
nvc++ -Minfo=stdpar -stdpar=multicore --c++17 -O4 vector_add.cpp

# Intel OneAPI
#dpcpp -O3 vector_add.cpp -ltbb
