// code from NVIDIA's CUDA C/C++ course
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void initWith(float num, float *a, int N){
    int i;
    for (i=0; i<N; ++i){
        a[i] = num;
    }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N){
    int i, idxWithinGrid, gridStride;
    idxWithinGrid = threadIdx.x + blockIdx.x * blockDim.x;
    gridStride = gridDim.x * blockDim.x;
    for (i=idxWithinGrid; i<N; i+=gridStride){
        result[i] = a[i]+b[i];
    }
}

void CheckElementsAre(float target, float *array, int N){
    int i;
    for (i=0; i<N;++i){
        if (array[i] != target){
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main(){
    const int N = 2<<20;
    size_t size = N*sizeof(float);

    float *a;
    float *b;
    float *c;

    checkCuda( cudaMallocManaged(&a, size) );
    checkCuda( cudaMallocManaged(&b, size) );
    checkCuda( cudaMallocManaged(&c, size) );

    initWith(4, a, N);
    initWith(5, b, N);
    initWith(0, c, N);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    CheckElementsAre(9, c, N);

    checkCuda( cudaFree(a) );
    checkCuda( cudaFree(b) );
    checkCuda( cudaFree(c) );

}