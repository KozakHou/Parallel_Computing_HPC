#include<iostream>
#include<thread>
#include<mutex>

std::mutex mtx1, mtx2;

void thread_func1(){
    std::lock_guard<std::mutex> lock1(mtx1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lock2(mtx2);
}

void thread_func2(){
    std::lock_guard<std::mutex> lock2(mtx2);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lock1(mtx1);
}

// thread_func1: locks mtx1, then tries to lock mtx2
// thread_func2: locks mtx2, then tries to lock mtx1 which makes a deadlock