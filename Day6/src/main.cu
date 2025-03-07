// Code for naive matrix-matrix multiplication
// Code from NVIDIA's CUDA course


#include <stdio.h>
#include <assert.h>
#include <cstdlib>

#include "multiplicationKernels.h"
#include "helper_functions.h"

int main(int argc, char *argv[])
{
  assert(argc==4);
  int r1 = atoi(argv[1]), r2 = atoi(argv[2]), c2 = atoi(argv[3]);

  float *h_a, *d_a, *h_b, *d_b, *h_c, *d_c, *c_check; // Allocate a solution matrix for both the CPU and the GPU operations
  dim3 dimA(r1,r2), dimB(r2,c2), dimC(r1,c2);

  int memSizeA = dimA.y*dimA.x*sizeof(float);
  int memSizeB = dimB.y*dimB.x*sizeof(float);
  int memSizeC = dimC.y*dimC.x*sizeof(float);

  // Allocate memory
  cudaMallocHost (&h_a, memSizeA);
  cudaMallocHost (&h_b, memSizeB);
  cudaMallocHost (&h_c, memSizeC);
  cudaMallocHost (&c_check, memSizeC);
  cudaMalloc (reinterpret_cast<void**>(&d_a), memSizeA);
  cudaMalloc (reinterpret_cast<void**>(&d_b), memSizeB);
  cudaMalloc (reinterpret_cast<void**>(&d_c), memSizeC);

  // Initialize memory; create 2D matrices
  InitRandom(h_a, dimA.x*dimA.y);
  InitRandom(h_b, dimB.x*dimB.y);
  InitWith(0, h_c, dimC.x*dimC.y);
  InitWith(0, c_check, dimC.x*dimC.y);

  // copy arrays to device
  cudaMemcpyAsync(d_a, h_a, memSizeA, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_b, h_b, memSizeB, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_c, h_c, memSizeC, cudaMemcpyHostToDevice);

  // compute result to check
  matrixMulCPU(h_a, h_b, c_check, dimA, dimB);

  // naive GPU version
  int threads = 16;
  dim3 threadsPerBlock(threads, threads);
  dim3 blocksInGrid( (dimC.x + threadsPerBlock.x -1) / threadsPerBlock.x, 
                      (dimC.y + threadsPerBlock.y -1) / threadsPerBlock.y);
  matrixMulGPU<<<threadsPerBlock, blocksInGrid>>>(d_a, d_b, d_c, dimA, dimB );
  cudaDeviceSynchronize();
  cudaMemcpy(h_c, d_c, memSizeC, cudaMemcpyDeviceToHost);
  checkMatrices(c_check, h_c, dimC);

  // if "success" time the kernel
  cudaEvent_t NaiveStart, NaiveStop;
  cudaEventCreate(&NaiveStart); cudaEventCreate(&NaiveStop);
  cudaEventRecord(NaiveStart);
  int nIter = 300;
  for (int iter=0; iter < nIter; ++iter){
    matrixMulGPU<<<threadsPerBlock, blocksInGrid>>>(d_a, d_b, d_c, dimA, dimB );
  }
  cudaEventRecord(NaiveStop); cudaEventSynchronize(NaiveStop);
  float NaiveMatMulTime;
  cudaEventElapsedTime(&NaiveMatMulTime, NaiveStart, NaiveStop);
  float TimePerNaiveMatMul = NaiveMatMulTime / nIter;
  printf("Naive Matrix Multplication Time: %f\n", TimePerNaiveMatMul);

  // free allocated memory
  cudaFree(h_a); cudaFree(d_a); cudaFree(h_b);
  cudaFree(d_b); cudaFree(h_c); cudaFree(d_c);
  cudaFree(c_check); cudaEventDestroy(NaiveStart); cudaEventDestroy(NaiveStop);  
  return 0;
}
