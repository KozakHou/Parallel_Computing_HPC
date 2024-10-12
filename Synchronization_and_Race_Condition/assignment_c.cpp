#include<iostream>
#include<thread>
#include<mutex>
#include<vector>

std::mutex mtx;

// use 2 thread push 1 to 1000 to the same vector :std::vector<int> 
// ensure that the vector size is 2000

void thread_push(std::vector<int> &v){
    for(int i = 1; i <= 1000; ++i){
        std::lock_guard<std::mutex> guard(mtx);
        v.push_back(i);
    }
}

int main(){
    std::vector<int> v;
    std::thread t1(thread_push, std::ref(v));
    std::thread t2(thread_push, std::ref(v));

    t1.join();
    t2.join();


    std::cout << "Size of vector: " << v.size() << std::endl;

    return 0;
}