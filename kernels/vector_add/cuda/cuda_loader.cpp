
// Test of CUDA functions for loading modules
//
// Compile with
// g++  cuda_loader.cpp  -lcuda
//
// Originally motivated by reproducing some problems with LLVM OpenMP offload
//

#include <cuda.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using std::string;

int main()
{
   string fname("test_offload-openmp-nvptx64-nvidia-cuda.cubin");
   //string fname("test_offload-openmp-nvptx64-nvidia-cuda.s");

   // variable name says ptxfile, but file can also be cubin, fatbin, etc.
   std::ifstream ptxfile(fname);
   std::stringstream buffer;
   buffer << ptxfile.rdbuf();
   ptxfile.close();

   string ptx_kernel = buffer.str();

   CUdevice cuDevice;
   CUcontext cuContext;
   cuInit(0);
   cuDeviceGet(&cuDevice, 0);
   cuCtxCreate(&cuContext, 0, cuDevice);

   const int nopt = 5;
   CUjit_option options[nopt];
   options[0] = CU_JIT_INFO_LOG_BUFFER;
   options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
   options[2] = CU_JIT_ERROR_LOG_BUFFER;
   options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
   options[4] = CU_JIT_LOG_VERBOSE;
   //options[4] = CU_JIT_TARGET;

   void* opt_vals[nopt];
   const int bufsize = 10000;
   char buf[bufsize];
   char err_buf[bufsize];

   opt_vals[0] = buf;
   opt_vals[1] = (void *)bufsize;
   opt_vals[2] = err_buf;
   opt_vals[3] = (void *)bufsize;
   opt_vals[4] = (void *)1;
   //opt_vals[4] = (void *)CU_TARGET_COMPUTE_60;

   CUmodule Module;

   CUresult ret = cuModuleLoadDataEx(&Module,ptx_kernel.c_str(), nopt, options, opt_vals);
   //CUresult ret = cuModuleLoad(&Module,fname.c_str());

   std::cout << "log buffer" << std::endl;
   std::cout << buf << std::endl;;
   std::cout << "-----------" << std::endl;
   std::cout << "err buffer" << std::endl;
   std::cout << err_buf << std::endl;;
   std::cout << "-----------" << std::endl;

   if (ret != CUDA_SUCCESS) {
     const char *sptr;
     cuGetErrorString(ret,&sptr);
     std::cout << "Error: " << sptr << std::endl;
   } else {
     std::cout << "Success " <<std::endl;
   }
   return 0;
}
