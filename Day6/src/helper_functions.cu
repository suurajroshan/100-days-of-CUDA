#include <cstdlib>
#include <stdio.h>

void InitRandom(float *a, int N){
    for (int i=0; i < N; ++i){
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void InitWith(float num, float *a, int N){
    for (int i=0; i<N; ++i){
        a[i] = num;
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

 // Compare the two answers to make sure they are equal
 void checkMatrices(float *c_cpu, float *c_gpu, dim3 dimC){
  bool error = false;
  float eps = 1e-6;
  for( int row = 0; row < dimC.x && !error; ++row )
    for( int col = 0; col < dimC.y && !error; ++col ){
        float abs_error = fabs(c_cpu[row * dimC.y + col] - c_gpu[row * dimC.y + col]);
        float rel_error = abs_error / c_cpu[row * dimC.y + col];
        if (  rel_error  > eps )
        {
            printf("FOUND ERROR at c[%d][%d]\n", row, col);
            error = true;
            break;
      }
    }
  if (!error)
    printf("Success!\n");
}