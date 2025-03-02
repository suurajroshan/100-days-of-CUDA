// The goal of this program is to multiple two matrices A,B and store them in C. 
// C = A * B
// Note: A and B are square matrices. This is a naive matrix-matrix multiplication.

// ./<FILE>.o `r1` `c1/r2` `c2`

#include <stdio.h>
#include <assert.h>

void initWith(float num, float *a, int N){
    for (int i=0; i<N; ++i){
        a[i] = num;
    }
}

void matMulNaiveHost(float *a, float* b, float* result, int r1, int r2, int c2){
    
    for (int i=0; i < r1; ++i){
        for (int j=0; j < r2; ++j){
            for (int k=0; k < c2; ++k){
                result[k+i*c2] += a[j+i*r2] * b[k+j*c2];
            }
        }
    }

}

int main(int argc, char *argv[]){
    assert(argc==4);
    int r1 = atoi(argv[1]), r2 = atoi(argv[2]), c2 = atoi(argv[3]);

    float *A, *B, *C;
    float naiveExecution;

    size_t sizeofA = r1 * r2 * sizeof(float);
    size_t sizeofB = r2 * c2 * sizeof(float);
    size_t sizeofC = r1 * c2 * sizeof(float);

    cudaMallocManaged(&A, sizeofA);
    cudaMallocManaged(&B, sizeofB);
    cudaMallocManaged(&C, sizeofC);

    cudaEvent_t startNaive, stopNaive;
    cudaEventCreate(&startNaive);
    cudaEventCreate(&stopNaive);

    initWith(4, A, r1*r2);
    initWith(5, B, r2*c2);
    initWith(0, C, r1*c2);

    cudaEventRecord(startNaive);
    matMulNaiveHost(A, B, C, r1, r2, c2);
    cudaEventRecord(stopNaive);

    // for (int i=1; i<r1*c2+1; ++i){
    //     printf("%0.0f ", C[i-1]);
    //     if ((i%c2) == 0){
    //         printf("\n");
    //         }
    // }
    // printf("\n");

    cudaEventElapsedTime(&naiveExecution, startNaive, stopNaive);
    printf("Elapsed time for Naive Implementation: %f\n", naiveExecution);
}