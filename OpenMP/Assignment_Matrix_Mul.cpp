#include<iostream>
#include<omp.h>

int main(){
    const int N = 3;
    int A[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int x[N] = {1, 2, 3};
    int b[N] = {0, 0, 0};


    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            b[i] += A[i][j] * x[j];
        }
    }

    std::cout << "Result of matrix-vector multiplication: " << b[0] << " " << b[1] << " " << b[2] << std::endl;
    return 0;
}