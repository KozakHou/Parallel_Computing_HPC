#include<iostream>
#include<omp.h>

int main(){
    const int N = 10;
    int a[N];

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++){
            a[i] = i*i;
            int thread_id = omp_get_thread_num();
            std::cout << "Hello from the thread" << thread_id << " computed a[" << i << "] = " << a[i] << std::endl;
        }
    }
    return 0;
}