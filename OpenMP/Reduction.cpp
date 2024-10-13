#include<iostream>
#include<omp.h>

int main(){
    const int N = 10;
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= N; i++){
        sum += i;
    }

    std::cout << "Sum from 1 to "<< N << " = " << sum << std::endl;
    return 0;

}