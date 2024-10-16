This repository aims to record the journey of becoming an HPC Software Engineer, using lecture content from Imperial College London's course "Patterns for Parallel Computing," along with ChatGPT-o1's evaluations of code and concepts.

The content includes: Parallelism & Concurrency, Threads & Processes, Distributed Computing, OpenMP, MPI, Slurm scripting, and Applications for High-Performance Computing in Python & C++.



## 1. Introduction to High-Performance Computing and Parallel Computing

### What is HPC ?
Spoiler: HPC allows large problems to be broken down into smaller parts and computed simultaneously, thereby speeding up the overall process.

### Why do we need HPC ?
* Speed: Execute computational tasks faster.
* Expansion: Handle larger datasets and more complex models.
* Efficiency: Optimize resource usage and reduce costs.

### Basic Concept for Computer architecture
* CPU
* Memory stage levels: register, cache (L1, L2, L3), RAM
* Storage: SSD, HDD

### Benchmarks
* FLOPS: Floating Point Operations Per Second
* Latency: Time taken for a job to complete
* Throughput: The number of jobs completed in a fixed time.


## 2. Basic of Parallel Computing

### Multithreading
Multithreading: Running multiple threads within a single process, sharing the same memory space.

### Multiprocessing
Multiprocessing: Running multiple processes, each independent and with its own separate memory space.


### Concurrency
**Definition**: Concurrency refers to a program's ability to handle multiple tasks at overlapping time intervals, although these tasks may not be executed simultaneously. Concurrency focuses more on the structure and design of the program, enabling it to effectively manage multiple tasks.

#### Features:
- **Task alternation**: On a single-core processor, tasks switch quickly, making it appear as if multiple tasks are running simultaneously.
- **Shared resources**: Multiple tasks may share the same resources, requiring synchronization mechanisms to avoid race conditions.
- **Focus on correctness**: Ensures that multiple tasks interact without causing errors.

#### Examples:
- Multi-threaded programming on a single-core CPU, achieving multitasking through time slicing.
- Operating systems manage multiple processes and threads via schedulers.

### Parallelism
**Definition**: Parallelism refers to the execution of multiple tasks simultaneously at the same point in time. This requires hardware support, such as multi-core processors or multiple processors.
1. Data Parallelism: Execute the same operation on each element in a dataset simultaneously.
2. Task Parallelism: Execute different tasks or functions simultaneously.
   
#### Features:
- **Simultaneous execution**: Multiple tasks are executed simultaneously on different processing units.
- **Performance enhancement**: Tasks are completed faster through parallel execution.
- **Focus on efficiency**: Maximizes resource utilization to improve throughput.

#### Examples:
- Multi-threaded programming on a multi-core CPU, where each thread runs on a different core.
- In distributed computing systems, tasks are allocated to different machines to run concurrently.

### Summary of Differences

|  | **Concurrency** | **Parallelism** |
|-----------------|-----------------|-----------------|
| **Concept** | Allows multiple tasks to overlap in time but not necessarily run simultaneously | Multiple tasks run simultaneously on actual hardware |
| **Implementation** | Achieved through programming and task management, possible on both single-core and multi-core systems | Requires hardware support, such as multi-core processors or multiple machines |
| **Focus** | Focuses on program correctness and maintainability, handling synchronization and resource sharing | Focuses on improving performance and efficiency, maximizing resource utilization |
| **Hardware Requirements** | No special hardware required; achievable on both single-core and multi-core systems | Requires multi-core processors or multi-processor systems |

### Further Understanding
- **Concurrency** is a programming concept:
  - Emphasizes how to design a program to handle multiple tasks.
  - Involves dealing with synchronization, locks, and deadlocks.
  - Example: Using asynchronous programming, coroutines, etc.
  
- **Parallelism** is an execution concept:
  - Focuses on leveraging hardware to execute multiple tasks simultaneously.
  - Involves task division and load balancing.
  - Example: Using multi-processors, multi-core systems, GPU acceleration, etc.
### Tips:

**Synchronous** = Thread will complete an action

**Blocking** = Thread will wait until action is completed

- Non-Blocking vs Blocking: Whether the thread will periodically poll for whether that task is complete, or whether it should wait for the task to complete before doing anything else

- Synchronous vs Asynchronous: Whether to execute the operation as initiated by the program or as a response to an event from the kernel.

#### Examples:
- Asynchronous + Non-Blocking: I/O

- Asynchronous + Blocking: Threaded atomics (demonstrated in 3.)

- Synchronous + Blocking: Standard computing

- Synchronous + Non-Blocking: Webservers where an I/O operation can be performed, but one never checks if the operation is completed.

## 3. Synchronization Mechanisms and Race Conditions

### What is a Race Condition?
**Race Condition**: A race condition occurs when multiple threads or processes access and modify shared resources simultaneously, and the final result depends on the order of execution.
- **Issue**: Race conditions can lead to data inconsistency, program crashes, or other unpredictable behavior.

### Why is Synchronization Needed?
- **Ensuring Data Consistency**: Protect shared resources to ensure that only one thread can modify them at a time.
- **Avoiding Deadlock**: Properly manage locks to prevent threads from waiting indefinitely for each other’s resources.

### Common Synchronization Tools

#### Mutex (Mutual Exclusion Lock)
- **Purpose**: Ensures that only one thread can access a shared resource at a time.
- **Feature**: Provides an exclusive locking mechanism.

#### Read-Write Lock
- **Purpose**: Allows multiple threads to read a resource simultaneously but only one thread to write at any given time.
- **Feature**: Improves performance in read-heavy applications.

#### Condition Variable
- **Purpose**: Allows a thread to wait for a certain condition to become true before continuing execution.
- **Feature**: Used in conjunction with a mutex to coordinate between threads.

#### Semaphore
- **Purpose**: Controls access to a limited number of resources.
- **Feature**: A counting-based synchronization mechanism, useful for implementing resource pools.

#### Barrier
- **Purpose**: Ensures that a group of threads waits at a specific point until all threads reach that point before continuing execution.
- **Feature**: Used to synchronize the progress of multiple threads.

### Deadlock and Avoidance

#### What is Deadlock?
**Definition**: Deadlock occurs when two or more threads are waiting for locks held by each other, causing all threads involved to be unable to proceed.

#### Conditions for Deadlock:
1. **Mutual Exclusion**: Resources cannot be shared.
2. **Hold and Wait**: A thread holds a resource and simultaneously waits for another resource.
3. **No Preemption**: Resources cannot be forcibly taken from a thread.
4. **Circular Wait**: A circular chain of threads exists, where each thread is waiting for a resource held by the next in the chain.

### How to avoid Deadlock ?
* Ensure that all threads acquire locks in the same order.
* Lock multiple mutexes simultaneously to avoid deadlock.

## 4. OpenMP Shared Memory Parallel Programming (C++)
### Introduction to OpenMP

#### What is OpenMP?
Open Multi-Processing (OpenMP) is a set of compiler directives, library routines, and environment variables used to implement shared memory multithreading parallel programming. By adding simple directives (Pragmas) to the code, developers can easily convert serial programs into parallel programs.

#### Advantages:
- **Ease of use**: Uses compiler directives, requiring minimal changes to the program structure.
- **Portability**: Supports multiple platforms and compilers.
- **Scalability**: Programs can automatically adjust based on the number of available processors.

### Checking Compiler Support for OpenMP

Common C++ compilers like GCC, Clang, and Intel C++ Compiler support OpenMP.

#### Adding the OpenMP Flag During Compilation

- **GCC and Clang**: Use the `-fopenmp` flag.

```bash
g++ -fopenmp your_program.cpp -o your_program
```

### Basic Syntax and Concept
#### Parallel Region
In OpenMP, a block of code intended for parallel execution is marked using the `#pragma omp parallel` directive. This tells the compiler to execute the following code in parallel using multiple threads
```cpp
#pragma omp parallel
{
    // Code for parallelsim
}
```

#### Work-Sharing Constructs
To avoid each thread executing the same task, OpenMP provides mechanisms for dividing tasks between threads. This can be done using the `#pragma omp for` directive to distribute iterations of a loop among multiple threads.
```cpp
#pragma omp for
for (int i = 0; i < N; ++i) {
    // Parallel Execution of Loop Body
}
```
another terms we can use `sections` to distributed different tasks to different threads.
```cpp
#pragma omp sections
{
    #pragma omp section
    {
        // First tasks
    }
    #pragma omp section
    {
        // Second tasks
    }
    // Can be several section.
}
```
or use `parallel for` to simplify the code.
```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    // Parallel Execution of Loop Body
}
```

### Variable Scope in OpenMP: `shared` and `private`

- **shared**: Variables are shared among all threads. Each thread can access and modify the same instance of the variable.

- **private**: Each thread has its own private copy of the variable. Modifications made by one thread do not affect the other threads.

```cpp
#pragma omp parallel shared(a) private(b)
{
    // code section
}
```
### Synchronization Mechanisms
####  Critical Section

**Purpose**: The critical section is used to protect a section of code, ensuring that only one thread can execute it at a time.

In OpenMP, the `#pragma omp critical` directive is used to define a critical section. This prevents multiple threads from executing the enclosed code simultaneously, thus avoiding race conditions.
```cpp
#pragma omp critical
{
    // critical code
}
```
#### Atomic Operation
Execure a single sentence of code for atomic operation is more efficient than critical section.
```cpp
#pragma omp atomic
// code for atomic operation
```
### Reduction
**Purpose**: Reduction is used to perform parallel reduction operations on shared variables, such as summation, multiplication, etc. It allows multiple threads to compute partial results and combine them into a final result efficiently.

In OpenMP, the `reduction` clause is used to specify the reduction operation.
```cpp
#pragma omp parallel for reduction(operation:variable)
for (int i = 0; i < N; ++i) {
    // operation for variable
}
```

supported arithmetic operation
`+`, `*`, `-`, `&`, `|`, `^`, `&&`, `||`

### Environment Variable and functions whil operating
1. Environment Variable: `OMP_NUM_THREAD`
```bash
export OMP_NUM_THREAD=4
```
2. Setting within the code
```cpp
#include<omp.h>
// some code
omp_set_num_threads(4);
```

3. Assign while compiling
```bash
g++ -fopenmp -DOMP_NUM_THREADS=4 program.cpp -o program
```

#### Syntax Clearification
`omp_get_num_threads()`
- **Purpose**: Retrieves the total number of threads in the current parallel region.

`omp_get_thread_num()`
- **Purpose**: Retrieves the unique thread number (ID) of the current thread within the parallel region.

### Benchmarking
```cpp
double start_time = omp_get_wtime();
// code snippet
double end_time = omp_get_wtime();
std::cout << "Elapsed time: " << end_time - start_time << " seconds." << std::endl;
```

### Common OpenMP Pitfalls and Considerations

#### Data Races
- **Issue**: Failure to correctly specify the scope of variables can lead to interference between threads.
- **Solution**: Explicitly declare variables as `private` or `shared` to avoid conflicts.

#### Loop Dependencies
- **Issue**: Dependencies within the loop body prevent parallelization.
- **Solution**: Refactor the code to eliminate dependencies, or avoid parallelizing such loops.

#### Excessive Synchronization
- **Issue**: Overusing synchronization mechanisms (e.g., critical sections, atomic operations) can cause significant performance degradation.
- **Solution**: Use synchronization only when necessary, and consider alternative approaches to minimize its use.

### Summary

#### Key Concepts:
- **Parallel Region**: Defines the code region for parallel execution.
- **Work-Sharing Constructs**: Distribute tasks among different threads.
- **Synchronization Mechanisms**: Ensure that threads safely access shared resources.
- **Variable Scope**: Clearly specify whether variables are `shared` or `private` to avoid data races.

#### Performance Optimization:
- Avoid excessive synchronization and minimize the use of critical sections.
- Use mechanisms like `reduction` to fully utilize parallel computing capabilities.

## 5. Distributed Memory Parallel Programming with MPI (C++)
### Introduction to MPI

#### What is MPI?
**MPI (Message Passing Interface)** is a widely-used communication protocol designed for message passing in distributed computing environments. It provides a set of APIs that allow programs to communicate and collaborate across multiple computing nodes, such as multiple computers or processors.

#### Why use MPI?
- **Scalability**: Suitable for systems ranging from a few nodes to thousands or even millions of nodes.
- **High Performance**: Offers low-latency and high-throughput communication capabilities.
- **Portability**: Available on various platforms and hardware architectures.

### Environment Setup

#### Installing MPI Implementations
Common MPI implementations include:
- **MPICH**: A high-performance and portable MPI implementation.
- **OpenMPI**: An open-source MPI implementation widely used in high-performance computing.

#### Installation & Compiling
On Linux: 
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```
On macOS:
```bash
brew install open-mpi
```

####
```bash
>> mpic++ program.cpp -o program
>> mpirun -np 4 ./program
```
* -np 4: activate 4 MPI processes

### MPI Basic Concepts

#### MPI Communication Model

- **Point-to-Point Communication**:
  - `MPI_Send`: Sends a message.
  - `MPI_Recv`: Receives a message.

- **Collective Communication**:
  - `MPI_Bcast`: Broadcasts a message to all processes.
  - `MPI_Reduce`: Performs a reduction operation (e.g., summing data from all processes).
  - `MPI_Barrier`: Synchronizes processes, ensuring all reach the same point before continuing.

#### MPI Processes and Ranks

- **MPI Processes**: An MPI program consists of multiple processes running in parallel.
- **Rank**: Each process has a unique identifier (rank) that starts from 0.

### MPI Basic Operation
```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize the MPI enviornment

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Retrieves the total number of processes running in the MPI communicator.

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Gets the rank (identifier) of the current process within the communicator.

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len); // acquire processes name

    std::cout << "Hello from processor " << processor_name << ", rank " << world_rank
              << " out of " << world_size << " processors" << std::endl;

    MPI_Finalize(); // end the MPI environment
    return 0;
}
```
`*MPI_Init`: Initializes the MPI environment. This function must be called before any other MPI function.
  
`MPI_Comm_size`: Retrieves the total number of processes in the communicator.

`MPI_Comm_rank`: Retrieves the rank (identifier) of the current process in the communicator.

`MPI_Finalize`: Terminates the MPI environment and releases any allocated resources.


### Point-to-Point Communication

#### `MPI_Send` and `MPI_Recv`

**MPI_Send**
```cpp
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
```

**MPI_Recv**
```cpp
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);

```

### Collective Communication

####  Broadcast (MPI_Bcast)
**MPI_Bcast**: Sends data from one process (the root process) to all other processes in a communicator.

```cpp
int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
```
`buf`: The starting address of the data to be broadcasted.
`count`: Number of elements in the buffer.
`datatype`: Data type of the elements in the buffer.
`root`: The rank of the process that is broadcasting the data.
`comm`: The communicator (e.g., MPI_COMM_WORLD).

### Reduce
**MPI_Reduce**: Performs a specified operation (e.g., summation, multiplication) on data from all processes and sends the result to the root process.

```cpp
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
```

`sendbuf`: Starting address of the send buffer (data to be reduced).
`recvbuf`: Starting address of the receive buffer (only valid at the root).
`count`: Number of elements in the buffer.
`datatype`: Data type of the elements.
`op`: Operation to be applied (e.g., MPI_SUM, MPI_PROD).
`root`: The rank of the root process where the result will be stored.
comm: The communicator (e.g., MPI_COMM_WORLD).

### Scatter and Gather

#### MPI_Scatter
**MPI_Scatter**: Distributes an array of data from the root process to all processes in the communicator, where each process receives a portion of the data.

#### Syntax:
```cpp
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
```
`sendbuf`: Address of the array to be scattered (valid only at the root).
`sendcount`: Number of elements sent to each process.
`sendtype`: Data type of the elements in the send buffer.
`recvbuf`: Address of the buffer where the scattered data will be received.
`recvcount`: Number of elements received by each process.
`recvtype`: Data type of the elements in the receive buffer.

### MPI Data Types and Topologies

#### MPI Data Types
- **Basic Data Types**:
  - `MPI_INT`
  - `MPI_FLOAT`
  - `MPI_DOUBLE`
  - `MPI_CHAR`
  - ...

- **Custom Data Types**:
  - You can define custom data structures using functions like `MPI_Type_create_struct`.

#### MPI Topologies
- **Purpose**: Defines the communication structure between processes, such as ring, grid, etc.

- **Functions**:
  - `MPI_Cart_create`: Creates a Cartesian topology.
  - `MPI_Graph_create`: Creates a graph topology.

---

### Non-blocking Communication

- **Advantages**: Non-blocking communication allows a process to continue performing other tasks while sending or receiving data, improving parallelism.

- **Non-blocking Functions**:
  - `MPI_Isend`: Initiates a non-blocking send operation.
  - `MPI_Irecv`: Initiates a non-blocking receive operation.

- **Wait Functions**:
  - `MPI_Wait`: Waits for a non-blocking operation to complete.
  - `MPI_Test`: Tests if a non-blocking operation has completed.

### MPI and Parallel Algorithm Design

1. **Problem Decomposition**: Break down the computational task into parts that can be executed in parallel.

2. **Data Distribution**: Distribute data among different processes to minimize communication overhead.

3. **Communication Patterns**: Optimize the communication between processes to avoid bottlenecks.

4. **Load Balancing**: Ensure that the computational workload is evenly distributed among processes to prevent some from being idle.

#### Performance Testing Tools:
- **MPI Built-in Timing Function**:
```cpp
  double MPI_Wtime();
```

### Common Issues and Debugging Techniques

1. **Deadlock**: Processes wait on each other, causing the program to halt.
   - **Solution**: Check communication pairs to ensure that `Send` and `Recv` operations are correctly matched.

2. **Data Inconsistency**: Incorrect data between processes.
   - **Solution**: Verify data types and communication parameters to ensure consistent data exchange.

3. **Using Debugging Tools**:
   - MPI provides debugging options that can help track issues.
   - Professional MPI debugging tools, such as **TotalView** and **Allinea DDT**, can assist in diagnosing complex issues in parallel programs.

### Summary

MPI is a powerful tool for implementing distributed memory parallel programming, particularly suited for multi-node computing environments.

#### Key Concepts:
- **Initialization and Finalization**: `MPI_Init` and `MPI_Finalize`.
- **Processes and Ranks**: Each MPI process has a unique rank.
- **Point-to-Point Communication**: `MPI_Send` and `MPI_Recv`.
- **Collective Communication**: `MPI_Bcast`, `MPI_Reduce`, and others.

#### Performance Optimization:
- Efficiently divide tasks and data to minimize communication overhead.
- Use non-blocking communication to overlap computation and communication.

#### Debugging and Problem Solving:
- Ensure proper communication matching to avoid deadlock.
- Use debugging tools and logs to troubleshoot issues.
