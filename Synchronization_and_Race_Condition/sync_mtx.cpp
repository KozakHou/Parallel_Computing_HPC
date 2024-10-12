#include<iostream>
#include<thread>
#include<mutex>

std::mutex mtx; // mutex for critical section

// void print_counter(int &counter){
//     for(int i = 0; i < 1000; ++i){
//         mtx.lock(); // lock the critical section
//         ++counter;
//         mtx.unlock(); // unlock the critical section
//     }
// }


void print_counter(int &counter){
    for(int i = 0; i < 1000; ++i){
        std::lock_guard<std::mutex> guard(mtx); // automatically lock and unlock the critical section in case of forgeting to unlock
        ++counter;
    }
}


int main(){
    int counter = 0;
    std::thread t1(print_counter, std::ref(counter));
    std::thread t2(print_counter, std::ref(counter));
    std::thread t3(print_counter, std::ref(counter));

    t1.join();
    t2.join();
    t3.join();

    std::cout << "Couter value: " << counter << std::endl;
    return 0;
}