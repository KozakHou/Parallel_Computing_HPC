#include<iostream>
#include<omp.h>

int main(){
    const int N = 1000000;
    double sum = 0.0;

    double start_time = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i){
        sum += 1.0 / (i + 1);
    }

    double end_time = omp_get_wtime();
    std::cout << "Sum = " << sum << std::endl;
    std::cout << "Time = " << end_time - start_time << std::endl;
    
    return 0;
}