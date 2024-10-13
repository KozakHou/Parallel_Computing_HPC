#include<iostream>
#include<omp.h>

int main(){
    int shared_var = 0;

    #pragma omp parallel private(shared_var)
    {
        shared_var = omp_get_thread_num();
        std::cout << "Thread " << omp_get_thread_num() << " has private shared_var = " << shared_var << std::endl; 
    }

    std::cout << "After parallel region, shared_var = " << shared_var << std::endl;
    return 0;

}
