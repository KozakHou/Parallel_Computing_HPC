#include<iostream>
#include<omp.h>

int main(){
    const int N = 1000000;
    double *a = new double[N];
    double sum = 0;

    // Initialize the array
    #pragma omp parallel for
    for (int i = 0; i < N; i++){
        a[i] = i;
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++){
        sum += a[i];
    }

    std::cout << "Sum from 1 to "<< N << " = " << sum << std::endl;

    delete[] a;
    return 0;
}