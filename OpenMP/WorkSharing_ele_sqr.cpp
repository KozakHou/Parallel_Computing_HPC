#include<iostream>
#include<omp.h>

int main(){
    const int N = 10;
    double *a = new double[N];

    // Initialize the array
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        a[i] = i;
    }

    // Compute the square of each element
    #pragma omp parallel for 
    for (int i = 0; i < N; i++){
        a[i] = a[i] * a[i];
    }

    for (int i = 0; i < N; ++i){
        std::cout << "a[" << i << "] = " << a[i] << std::endl;
    }

    delete[] a;
    return 0;
}