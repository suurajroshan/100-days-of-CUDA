// The goal of this program is to multiple two matrices A,B and store them in C. 
// C = A * B
// Note: A and B are square matrices. This is a naive matrix-matrix multiplication.

// ./<file>.o `r1` `c1/r2` `c2`

#include <stdio.h>
#include <assert.h>

void initWith(float num, float *a, int N){
    for (int i=0; i<N; ++i){
        a[i] = num;
    }
}

void matMulSerial(float *a, float* b, float* result, int xdim, int ydim){
    
    for (int i=0; i < )
}

int main(int argc, char *argv[]){
    assert(argc==4);
    int xdim = atoi(argv[1]), ydim = atoi(argv[2]);

    float *A, *B, *C;
    int numberOfElements = xdim * ydim;
    size_t size = sizeof(numberOfElements);

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    initWith(4, A, numberOfElements);
    initWith(5, B, numberOfElements);

    for (int i=1; i<numberOfElements+1; ++i){
        printf("%0.0f ", A[i-1]);
        if ((i%xdim) == 0){
            printf("\n");
            }
    }
    printf("\n");
}