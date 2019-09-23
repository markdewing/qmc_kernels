
// Implementation of vector add in OpenCL


// stop deprecation warnings about clCreateCommandQueue
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>
#include <stdio.h>
#include "timer.h"


#define RealType float

void check_status(int status, const char *api_name)
{
  if (status != CL_SUCCESS) {
    printf("%s failed\n",api_name);
  }
}

int main()
{
  int n = 10000;

  RealType *h_a = (RealType *)malloc(sizeof(RealType)*n);
  RealType *h_b = (RealType *)malloc(sizeof(RealType)*n);
  RealType *h_c = (RealType *)malloc(sizeof(RealType)*n);

  for (int i = 0; i < n; i++) {
    h_a[i] = 1.0;
    h_b[i] = 1.0*i;
  }

  cl_platform_id cpPlatform[3];
  cl_int err = clGetPlatformIDs(3, cpPlatform, NULL);

  check_status(err, "clGetPlatformIDs");

  int platform_id_index = 1;

#define NAME_STR_LEN 100
  char platform_name[NAME_STR_LEN];
  err = clGetPlatformInfo(cpPlatform[platform_id_index], CL_PLATFORM_NAME, NAME_STR_LEN, platform_name, NULL);
  check_status(err, "clGetPlatformInfo");

  char vendor_name[NAME_STR_LEN];
  err = clGetPlatformInfo(cpPlatform[platform_id_index], CL_PLATFORM_VENDOR, NAME_STR_LEN, vendor_name, NULL);
  check_status(err, "clGetPlatformInfo");

  printf("Platform name : %s\n",platform_name);
  printf("Platform vendor : %s\n",vendor_name);

  cl_device_id device_id;
  err = clGetDeviceIDs(cpPlatform[platform_id_index], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
  check_status(err, "clGetDeviceIDs");

  char device_name[NAME_STR_LEN];
  err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, NAME_STR_LEN, device_name, NULL);
  check_status(err, "clGetDeviceInfo");

  printf("Device name : %s\n",device_name);

  cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  check_status(err, "clCreateContext");

  cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

  check_status(err, "clCreateCommandQueue");


  int max_kernel_string_size = 10000;
  char *kernel_str = (char *)malloc(max_kernel_string_size);
  FILE *f = fopen("kernel_vector_add.cl","r");
  if (f == NULL) {
    printf("Opening kernel_vector_add.cl failed\n");
  }
  size_t nread = fread(kernel_str, 1, max_kernel_string_size, f);
  kernel_str[nread] = '\0';
  fclose(f);

  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_str,  NULL, &err);

  check_status(err, "clCreateProgramWithSource");

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("clBuildProgram failed\n");
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log_msg = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log_msg, NULL);
    printf("Build log \n%s\n",log_msg);

  }

  cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
  check_status(err, "clCreateKernel");


  cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, n*sizeof(RealType), NULL, NULL);
  cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, n*sizeof(RealType), NULL, NULL);
  cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*sizeof(RealType), NULL, NULL);


  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, n*sizeof(RealType), h_a, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, n*sizeof(RealType), h_b, 0, NULL, NULL);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);


  double host_start = clock();
  size_t local_size = 64;
  size_t global_size = n/local_size + 1;
  cl_event kernel_event;
  //printf("local size = %d  global_size = %d\n",local_size, global_size);
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (size_t *)&n,  NULL, 0, NULL, &kernel_event);
  check_status(err, "clEnqueueNDRangeKernel");

  clWaitForEvents(1, &kernel_event);

  clFinish(queue);

  double host_end = clock();

  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, n*sizeof(RealType), h_c, 0, NULL, NULL);
  printf("h_c[0] = %g\n",h_c[0]);
  printf("h_c[n-1] = %g\n",h_c[n-1]);

  //for (int i = 0; i < n; i++) {
  //  printf("h_c[%d] = %g\n",i,h_c[i]);
  //}
  
  printf("host time = %g us\n",(host_end-host_start)*1e6);

  cl_ulong kernel_start;
  cl_ulong kernel_end;

  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(kernel_start), &kernel_start, NULL);
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(kernel_end), &kernel_end, NULL);

  printf("kernel time = %g us\n",(kernel_end-kernel_start)/1.0e3);


  // Free resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
