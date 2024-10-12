#include<iostream>
#include<thread>


// Create 5 threads, and each thread print 1 to 5
void print_message(const std::string& message){

    for(int i = 1; i <= 5; i++){
        std::cout << "Thread ID: " << std::this_thread::get_id() << " - Message: " << message << " - " << i << std::endl;
    }
}

int main(){
    std::thread thread1(print_message, "Hello from thread 1");
    std::thread thread2(print_message, "Hello from thread 2");
    std::thread thread3(print_message, "Hello from thread 3");
    std::thread thread4(print_message, "Hello from thread 4");
    std::thread thread5(print_message, "Hello from thread 5");

    thread1.join();
    thread2.join();
    thread3.join();
    thread4.join();
    thread5.join();

    return 0;
}