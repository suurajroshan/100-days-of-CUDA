#include "multiplicationKernels.h"
#include <stdio.h>

__global__ void matrixMulGPU( float * a, float * b, float * c, dim3 dimA, dim3 dimB )
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    dim3 dimC(dimA.x, dimB.y);
    float val = 0;
    if (row < dimC.x && col < dimC.y){
        for (int i = 0; i < dimA.y; ++i){
            val += a[i + row*dimA.y] * b[col + i*dimB.y];
        }
        c[row * dimC.y + col] = val;
    }
}

void matrixMulCPU( float * a, float * b, float * c, dim3 dimA, dim3 dimB )
{
    for (int i = 0; i < dimA.x; ++i){
        for (int k = 0; k < dimB.y; ++k){
            float val = 0;
            for (int j = 0;  j < dimA.y; ++j){
                val += a [i*dimA.y + j] * b[k + j*dimB.y];
            }
            c[k+i*dimB.y] = val;
        }
    }
}