// https://www.youtube.com/watch?v=Iz6feoh9We8
// https://github.com/yohanesgultom/parallel-programming-assignment/blob/master/PR2/opencl/device_query.c

#include <stdint.h>
#include <cassert>
#include "CL/opencl.h"
#include "myutils.hpp"

#define ATTRIBUTE_COUNT 5
#define VEC_SIZE 256

const cl_platform_info attributeTypes[ATTRIBUTE_COUNT] = {
  CL_PLATFORM_NAME,
  CL_PLATFORM_VENDOR,
  CL_PLATFORM_VERSION,
  CL_PLATFORM_PROFILE,
  CL_PLATFORM_EXTENSIONS
};

const char* const attributeNames[ATTRIBUTE_COUNT] = {
  "CL_PLATFORM_NAME",
  "CL_PLATFORM_VENDOR",
  "CL_PLATFORM_VERSION",
  "CL_PLATFORM_PROFILE",
  "CL_PLATFORM_EXTENSIONS"
};

int main() {
  cl_platform_id platforms[64];
  cl_uint platformCount;
  cl_int platformsResult = clGetPlatformIDs(64, platforms, &platformCount);
  assert(platformsResult == CL_SUCCESS);

  for (int i = 0; i < platformCount; i++) {
    for (int j = 0; j < ATTRIBUTE_COUNT; j++) {
      // Get platform attribute value size
      size_t infosize;
      assert(clGetPlatformInfo(platforms[i], attributeTypes[j], 0, nullptr, &infosize) == CL_SUCCESS);
      char info[infosize];

      // Get platform attribute value
      assert(clGetPlatformInfo(platforms[i], attributeTypes[j], infosize, info, nullptr) == CL_SUCCESS);
      printf("%d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
    }
  }

  printf("\n");

  cl_device_id device = nullptr;
  for (int i = 0; i < platformCount; i++) {
    cl_device_id devices[64];
    cl_uint deviceCount;
    cl_int deviceResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);

    if (deviceResult == CL_SUCCESS) {
      for (int j = 0; j < deviceCount; j++) {
        char vendorName[256];
        size_t vendorNameLength;
        cl_int deviceInfoResult = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength);
        if (deviceInfoResult == CL_SUCCESS) {
          device = devices[j];
          print("Using device(vendor name): " + std::string(vendorName) + "\n");
          break;
        }
      }
    }
  }

  assert(device);

  cl_int contextResult;
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &contextResult);
  assert(contextResult == CL_SUCCESS);

  cl_int commandQueueResult;
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &commandQueueResult);
  assert(commandQueueResult == CL_SUCCESS);

  std::string clFile = readFromFile("../../src/vecsum.cl");
  const char* programSource = clFile.c_str();
  size_t length = 0;
  cl_int programResult;
  cl_program program = clCreateProgramWithSource(context, 1, &programSource, &length, &programResult);
  assert(programResult == CL_SUCCESS);

  cl_int programBuildResult = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
  if (programBuildResult != CL_SUCCESS) {
    char log[256];
    size_t logLength;
    cl_int programBuildInfoResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 256, log, &logLength);
    assert(programBuildInfoResult == CL_SUCCESS);
    print(log);
    assert(log);
  }

  cl_int kernelResult;
  cl_kernel kernel = clCreateKernel(program, "vector_sum", &kernelResult);
  assert(kernelResult == CL_SUCCESS);

  float vecData_a[VEC_SIZE];
  float vecData_b[VEC_SIZE];

  for (int i = 0; i < VEC_SIZE; i++) {
    vecData_a[i] = Random::GetFloat(-50.f, 50.f);
    vecData_b[i] = Random::GetFloat(-50.f, 50.f);
  }

  cl_int vecResult_a;
  cl_mem vec_a = clCreateBuffer(context, CL_MEM_READ_ONLY, VEC_SIZE * sizeof(float), nullptr, &vecResult_a);
  assert(vecResult_a == CL_SUCCESS);

  cl_int enqueueVecResult_a = clEnqueueWriteBuffer(queue, vec_a, CL_TRUE, 0, VEC_SIZE * sizeof(float), vecData_a, 0, nullptr, nullptr);
  assert(enqueueVecResult_a == CL_SUCCESS);

  cl_int vecResult_b;
  cl_mem vec_b = clCreateBuffer(context, CL_MEM_READ_ONLY, VEC_SIZE * sizeof(float), nullptr, &vecResult_b);
  assert(vecResult_b == CL_SUCCESS);

  cl_int enqueueVecResult_b = clEnqueueWriteBuffer(queue, vec_b, CL_TRUE, 0, VEC_SIZE * sizeof(float), vecData_b, 0, nullptr, nullptr);
  assert(enqueueVecResult_b == CL_SUCCESS);

  cl_int vecResult_c;
  cl_mem vec_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VEC_SIZE * sizeof(float), nullptr, &vecResult_c);
  assert(vecResult_c== CL_SUCCESS);

  cl_int kernelArgResult_a = clSetKernelArg(kernel, 0, sizeof(cl_mem), &vec_a);
  cl_int kernelArgResult_b = clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_b);
  cl_int kernelArgResult_c = clSetKernelArg(kernel, 2, sizeof(cl_mem), &vec_c);

  assert(kernelArgResult_a == CL_SUCCESS);
  assert(kernelArgResult_b == CL_SUCCESS);
  assert(kernelArgResult_c == CL_SUCCESS);

  size_t globalWorkSize = VEC_SIZE;
  size_t localWorkSize = 64;
  cl_int enqueueKernelResult = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
  assert(enqueueKernelResult == CL_SUCCESS);

  float vecData_c[VEC_SIZE];
  cl_int enqueueReadBufferResult = clEnqueueReadBuffer(queue, vec_c, CL_TRUE, 0, VEC_SIZE * sizeof(float), vecData_c, 0, nullptr, nullptr);
  assert(enqueueReadBufferResult == CL_SUCCESS);

  clFinish(queue);

  std::cout << "Result: ";
	for ( int i = 0; i < VEC_SIZE; ++i )
    std::cout << vecData_c[i] << std::endl;

  clReleaseMemObject(vec_a);
	clReleaseMemObject(vec_b);
	clReleaseMemObject(vec_c);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);

  return 0;
}

