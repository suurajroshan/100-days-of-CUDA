// The goal of this program is to multiple two matrices A,B and store them in C. 
// C = A * B
// Note: A and B are square matrices. This is a naive matrix-matrix multiplication.

// ./<FILE>.o `r1` `c1/r2` `c2`

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <stdlib.h>

void initWith(float num, float *a, int N){
    for (int i=0; i<N; ++i){
        a[i] = num;
    }
}

void InitRandom(float *a, int N){
    for (int i=0; i < N; ++i){
        a[i] = (float)rand() / (float)RAND_MAX;
    }
}

void printMatrix(float *M,int m1, int m2){
    for (int i=1; i<m1*m2+1; ++i){
        printf("%f ", M[i-1]);
        if ((i%m2) == 0){
            printf("\n");
            }
    }
    printf("\n");
}

__global__
void dotProduct(float *x, int xlen, float *y, int ylen, float *sum){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    // int j = threadIdx.y+blockDim.y+blockDim.y;
    sum[i] += x[i]*y[i];
}

void vectorSum(float* vector, int vectorLength, float* result){
    for (int i=0; i<vectorLength; ++i){
        *result += vector[i];
    }
}

// matrix multiplication kernel using dot products of rows of A with coloumns of B
void MatrixMultiplication(float *A, int sizeAy, int sizeAx, float *B, int sizeBy, int sizeBx, 
float* C){
    float *d_A, *d_B, *d_C;

    unsigned int memSizeA = sizeAy*sizeAx*sizeof(float);
    unsigned int memSizeB = sizeBy*sizeBx*sizeof(float);
    unsigned int memSizeC = sizeAy*sizeBx*sizeof(float);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMalloc((void**)&d_A, sizeAy*sizeAx*sizeof(float));
    cudaMalloc((void**)&d_B, sizeBy*sizeBx*sizeof(float));
    cudaMalloc((void**)&d_C, sizeAy*sizeBx*sizeof(float));

    cudaMemcpyAsync(d_A, A, memSizeA, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, memSizeB, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, C, memSizeC, cudaMemcpyHostToDevice, stream);

    float *cVec;
    int memSizeC_vec = sizeAx*sizeof(float);
    cudaMalloc((void**)&cVec, memSizeC_vec);
    float *h_cVec;
    cudaMallocHost((void**)&h_cVec, memSizeC_vec);
    

    for (int i=0, j=0; i<sizeAy*sizeAx && j<sizeBy*sizeBx; i+=sizeAx, j+=sizeBx){
            float *bVec = &d_B[j];
            float *aVec = &d_A[i];
            printf("%d, %d\n", i, j );
            dim3 dimBlock(sizeAx, sizeAy, 1);
            dotProduct<<<1, sizeAx, 0 , stream>>>(aVec, sizeAx, bVec, sizeBy, cVec);
            cudaMemcpyAsync(h_cVec, cVec, memSizeC_vec, cudaMemcpyDeviceToHost, stream);
            vectorSum( h_cVec, sizeAx, &C[i+j*sizeAx] );
    }
    cudaDeviceSynchronize();
}



int main(int argc, char *argv[]){
    assert(argc==4);
    int r1 = atoi(argv[1]), r2 = atoi(argv[2]), c2 = atoi(argv[3]);

    float *h_A, *h_B, *h_C;

    size_t sizeA = r1*r2*sizeof(float);
    size_t sizeB = r2*c2*sizeof(float);
    size_t sizeC = r1*c2*sizeof(float);

    cudaMallocHost((void**)&h_A, sizeA);
    cudaMallocHost((void**)&h_B, sizeB);
    cudaMallocHost((void**)&h_C, sizeC);

    // InitRandom(h_A, r1*r2);
    // InitRandom(h_B, r2*c2);
    initWith(5, h_A, r1*r2);
    initWith(4, h_B, r2*c2);
    initWith(0, h_C, r1*c2);

    MatrixMultiplication(h_A, r1, r2, h_B, r2, c2, h_C);

    // cudaEvent_t startNaive, stopNaive;
    // cudaEventCreate(&startNaive);
    // cudaEventCreate(&stopNaive);

    // cudaEventRecord(startNaive);
    // matMulNaiveHost(A, B, C, r1, r2, c2);
    // cudaEventRecord(stopNaive);

    printMatrix(h_C, r1, c2);

    // cudaEventElapsedTime(&naiveExecution, startNaive, stopNaive);
    // printf("Elapsed time for Naive Implementation: %f\n", naiveExecution);
}