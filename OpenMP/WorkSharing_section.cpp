#include<iostream>
#include<omp.h>

int main(){
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                std::cout << "Task A executed by thread " << omp_get_thread_num() << std::endl;
            }
            #pragma omp section
            {
                std::cout << "Task B executed by thread " << omp_get_thread_num() << std::endl;
            }
        }
    }
    return 0;
}