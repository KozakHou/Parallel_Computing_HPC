#include<iostream>
#include<omp.h>

int main(){
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from the thread" << thread_id << std::endl;
    }
    return 0;
}

// compile with: g++ -fopenmp Parallelism.cpp -o Parallelism