#include<iostream>
#include<thread>
#include<atomic>


std::atomic<int> counter(0); // atomic variable

void increment_counter(){
    for(int i = 0; i < 1000; ++i){
        ++counter;
    }
}


int main(){
    std::thread t1(increment_counter);
    std::thread t2(increment_counter);
    std::thread t3(increment_counter);

    t1.join();
    t2.join();
    t3.join();

    std::cout << "Couter value: " << counter << std::endl;
    return 0;
}