// The goal of this program is to multiple two matrices A,B and store them in C. 
// C = A * B
// Note: A and B are square matrices. This is a naive matrix-matrix multiplication.

// ./<FILE>.o `r1` `c1/r2` `c2`

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <stdlib.h>
#include <cuda_runtime_api.h>

void initWith(float num, float *a, int N){
    for (int i=0; i<N; ++i){
        a[i] = num;
    }
}

template<int BLOCK_SIZE> __global__
void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB){
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = wb * BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;
    for (int a = aBegin, b = bBegin; 
        a<=aEnd;
        a += aStep, b += bStep){
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
            As[ty][tx] = A[a + wA * ty + tx];
            Bs[ty][tx] = B[b + wB * ty + tx];

            __syncthreads();

            #pragma unroll
            for(int k=0; k < BLOCK_SIZE; ++k){
                Csub += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();

            int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
            C[c + wB * ty + tx] = Csub;
        }
}

int MatrixMultiply(int block_size, const dim3 &dimsA, const dim3 &dimB){
    // allocate host memory
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int memSizeA = sizeof(float) * size_A;
    float *h_A;
    cudaMallocHost(&h_A, memSizeA);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int memSizeB = sizeof(float) * size_B;
    float *h_B;
    cudaMallocHost(&h_B, memSizeB);

    dim3 dimC(dimB.x, dimA.y, 1);
    unsigned int memSizeC = dimC.x * dimC.y * sizeof(float);
    float *h_C;
    cudaMallocHost(&h_C, memSizeC);

    // initialize host memory
    const float valB = 0.01f;
    initWith(5.0f, h_A, size_A);
    initWith(4.0f, h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(reinterpret_cast<void**>(&d_A), memSizeA);
    cudaMalloc(reinterpret_cast<void**>(&d_B), memSizeB);
    cudaMalloc(reinterpret_cast<void**>(&d_C), memSizeC);

    // copy to device
    cudaMemcpyAsync(d_A, h_A, memSizeA, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, h_B, memSizeB, cudaMemcpyHostToDevice);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    if (block_size == 16){
        MatrixMulCUDA<16><<<grid, threads>>>(d_c, d_A, d_B, dimsA.x, dimsB.x);
    }
    else {
        MatrixMulCUDA<32><<<grid, threads>>>(d_c, d_A, d_B, dimsA.x, dimsB.x);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
        MatrixMulCUDA<16>
            <<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        } else {
        MatrixMulCUDA<32>
            <<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    cudaEventCreate(stop);
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                                    static_cast<double>(dimsA.y) *
                                    static_cast<double>(dimsB.x);
     double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
     printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
            " WorkgroupSize= %u threads/block\n",
            gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
    
    bool correct = true;
    double eps = 1.e-6;

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
             i, h_C[i], dimsA.x * valB, eps);
      correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
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

// matrix multiplication kernel using dot products of rows of A with coloumns of B
void MatrixMultiplication(float *A, dim3 dimA, float *B, dim3 dimB, float* C, dim3 dimC){
    float *d_A, *d_B, *d_C;

    unsigned int memSizeA = dimA.x*dimA.y*sizeof(float);
    unsigned int memSizeB = dimB.x*dimB.y*sizeof(float);
    unsigned int memSizeC = dimA.y*dimB.x*sizeof(float);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaMalloc((void**)&d_A, memSizeA);
    cudaMalloc((void**)&d_B, memSizeB);
    cudaMalloc((void**)&d_C, memSizeC);

    cudaMemcpyAsync(d_A, A, memSizeA, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, memSizeB, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, C, memSizeC, cudaMemcpyHostToDevice, stream);

    float *cVec;
    int memSizeC_vec = dimB.x*sizeof(float);
    cudaMalloc((void**)&cVec, memSizeC_vec);
    float *h_cVec;
    cudaMallocHost((void**)&h_cVec, memSizeC_vec);

    float *bVec;
    float *aVec;

    dotProduct<<<dimA.x, dimB.y, 0, stream>>>(d_A, dimA, d_B, dimB, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, memSizeC, cudaMemcpyDeviceToHost);

    // printMatrix(C, dimC.x, dimC.y);

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

    dim3 dimA(r1, r2, 1);
    dim3 dimB(r2, c2, 1);
    dim3 dimC(r2, c2, 1);

    MatrixMultiplication(h_A, dimA, h_B, dimB, h_C, dimC);

    // cudaEvent_t startNaive, stopNaive;
    // cudaEventCreate(&startNaive);
    // cudaEventCreate(&stopNaive);

    // cudaEventRecord(startNaive);
    // matMulNaiveHost(A, B, C, r1, r2, c2);
    // cudaEventRecord(stopNaive);

    // printMatrix(h_C, r1, c2);

    // cudaEventElapsedTime(&naiveExecution, startNaive, stopNaive);
    // printf("Elapsed time for Naive Implementation: %f\n", naiveExecution);
}