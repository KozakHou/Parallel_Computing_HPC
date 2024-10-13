#include<iostream>
#include<omp.h>

int main(){
    int counter = 0;

    #pragma omp parallel
    {
        for (int i = 0; i < 1000; ++i){
            #pragma omp atomic
            counter++;
        }
    }

    std::cout << "Counter = " << counter << std::endl;
    return 0;
}