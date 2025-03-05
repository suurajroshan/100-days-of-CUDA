__global__ void matrixMulGPU( int * a, int * b, int * c )
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.x;
    
    for (int i = 0; i < N; ++i){
    c[row*N+col] += a[row*N+i]*b[i*N+col];
    }
}