// This code is adaopted from cuda-samples repo

#include <stdio.h>
#include <assert.h>



void MatrixMultiply(float *a, float *b, const dim3& dimA, const dim3& dimB){
    a[block_size][block_size];
}


int main(){

    int block_size = 32;
    dim3 dimA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    MatrixMultiply(A, B, dimA, dimB)
}