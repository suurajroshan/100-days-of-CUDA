#pragma once
#include <cuda_runtime.h>

// runs of CPU
void matrixMulCPU( float * a, float * b, float * c, dim3 dimA, dim3 dimB );

// Naive approach of GPU
__global__ void matrixMulGPU( float * a, float * b, float * c, dim3 dimA, dim3 dimB );

void checkMatrices(float *c_cpu, float *c_gpu, dim3 dimC);
