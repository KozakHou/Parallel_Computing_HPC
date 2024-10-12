#include<iostream>
#include<thread>

void print_message(const std::string& message){
    std::cout << "Thread ID: " << std::this_thread::get_id() << " - Message: " << message << std::endl;
}


int main(){
    std::thread thread1(print_message, "Hello from thread 1");
    std::thread thread2(print_message, "Hello from thread 2");

    thread1.join();
    thread2.join();

    return 0;
}


// compile: g++  multi_thread.cpp -o multi_thread -pthread
// run: ./multi_thread
