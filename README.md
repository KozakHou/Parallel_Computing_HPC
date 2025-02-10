This repository aims to record the journey of mastering HPC, using lecture content from Imperial College London's course "Patterns for Parallel Computing," along with ChatGPT-o1's evaluations of code and concepts.

The content includes: Parallelism & Concurrency, Threads & Processes, OpenMP, MPI, Slurm scripting, and Applications for High-Performance Computing in Python & C++.



## 1. Introduction to High-Performance Computing and Parallel Computing

### What is HPC ?
Spoiler: HPC allows large problems to be broken down into smaller parts and computed simultaneously, thereby speeding up the overall process.

### Why do we need HPC ?
* Speed: Execute computational tasks faster.
* Expansion: Handle larger datasets and more complex models.
* Efficiency: Optimize resource usage and reduce costs.

### Basic Concept for Computer Architecture
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
`MPI_Init`: Initializes the MPI environment. This function must be called before any other MPI function.
  
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

### 6. Parallel Computing in Python

#### Multithreading in Python

##### `threading` Module
- The `threading` module is part of Python's standard library and provides support for multithreading, allowing concurrent execution of tasks.

##### Global Interpreter Lock (GIL)
- **What is GIL?**
  - The Global Interpreter Lock (GIL) is a mechanism in CPython (the most common Python interpreter) that ensures only one native thread executes Python bytecode at a time.

- **Impact of GIL**:
  - **CPU-bound tasks**: Multithreading cannot achieve true parallelism, and performance gains are limited.
  - **I/O-bound tasks**: Multithreading can effectively improve performance, as threads can switch while waiting for I/O operations to complete.

### Multiprocessing in Python

#### `multiprocessing` Module
- The `multiprocessing` module provides the ability to create multiple processes. Each process runs its own instance of the Python interpreter, allowing it to bypass the Global Interpreter Lock (GIL).
  
  By using multiple processes, you can achieve true parallelism (bypass the GIL), particularly for CPU-bound tasks.

### `concurrent.futures` Module

#### Introduction
- **`concurrent.futures`**: A module in Python's standard library (introduced in Python 3) that provides a high-level interface for concurrently executing tasks. It simplifies the implementation of multithreading and multiprocessing.

#### `ThreadPoolExecutor` and `ProcessPoolExecutor`
- **`ThreadPoolExecutor`**: A thread-based executor, suitable for I/O-bound tasks where waiting on I/O operations dominates execution time.
  
- **`ProcessPoolExecutor`**: A process-based executor, suitable for CPU-bound tasks that require parallel execution to fully utilize the CPU cores, bypassing the Global Interpreter Lock (GIL).

### MPI in Python

#### `mpi4py` Module
- **`mpi4py`**: A Python interface for MPI, providing an API similar to MPI's C/C++ interface, allowing MPI functionality to be used in Python programs.

#### Installing `mpi4py`
- **Prerequisite**: You need to have an MPI implementation installed, such as **MPICH** or **OpenMPI**.

- **Installation**:
  To install `mpi4py`, you can use `pip` after ensuring that MPI is installed on your system:
  ```bash
  pip install mpi4py
  ```
#### Basic usage
```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from rank {rank} out of {size} processors)
```
Run the code
```bash
mpirun -np 4 python mpi_hello.py
```

### Point-to-Point Communication

#### Example: `send` and `recv`

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'key1': [7, 2.72, 2+3j],
            'key2': ('abc', 'xyz')}
    comm.send(data, dest=1, tag=11)
    print("Process 0 sent data to Process 1")
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print("Process 1 received data:", data)
```

In this example: 

* Process 0 sends a dictionary to Processor 1 using comm.send()
* Process 1 recieves the data using comm.recv()

### Collective Communication
Example: Broadcast

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'a': 7, 'b': 3.14}
else:
    data = None

data = comm.bcast(data, root=0)
print(f"Process {rank} received data: {data}")
```
In this example:

* Process 0 initializes a dictionary and broadcasts it to all processes.
* All processes, including Process 0, receive the broadcasted data using comm.bcast().

Example: Reduce
```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

local_sum = rank + 1  # Each process computes its local value
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum = {global_sum}")
```
In this example:

* Each process computes a local sum (rank + 1).
* Process 0 collects and sums the local values from all processes using comm.reduce() with the MPI.SUM operation.

### Asynchronous Programming in Python

#### `asyncio` Module
- **`asyncio`**: An asynchronous I/O framework introduced in Python 3.4, based on coroutines and event loops.

#### Basic Concepts

- **Coroutine**: A function defined using `async def`, which can be suspended and resumed during its execution.
  
- **`await` keyword**: Used to wait for a coroutine or asynchronous task to complete.

- **Event Loop**: A mechanism that coordinates the execution of coroutines and ensures they are run in a non-blocking manner.

Example: Download url contents via asynchronous
```python
import asyncio
import aiohttp

urls = [
    'https://www.example.com',
    'https://www.python.org',
    'https://www.openai.com',
    # can add more links
]

async def download_content(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"Download completed：{url}，status：{response.status}")
            return await response.text()

async def main():
    tasks = [download_content(url) for url in urls]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

### Efficient I/O Operations in Python

- **`asyncio`**: Well-suited for concurrent execution of tasks that involve a large number of I/O operations. It allows for non-blocking, asynchronous I/O, making it highly efficient for tasks like network requests, file operations, and database access.

- **`aiohttp` Module**: A third-party library that supports asynchronous HTTP requests, built on top of `asyncio`. It enables efficient handling of large numbers of HTTP requests in an asynchronous, non-blocking manner.

### Choosing the Right Parallel Method

#### I/O-bound Tasks
- **Recommended Methods**: `threading`, `asyncio`, `concurrent.futures.ThreadPoolExecutor`.
- **Reason**: The Global Interpreter Lock (GIL) has minimal impact on I/O operations, allowing these methods to effectively improve performance for tasks such as network requests, file I/O, and database operations.

#### CPU-bound Tasks
- **Recommended Methods**: `multiprocessing`, `concurrent.futures.ProcessPoolExecutor`.
- **Reason**: These methods bypass the GIL, enabling true parallelism and fully utilizing multi-core CPUs for tasks like heavy computation or data processing.

#### Distributed Computing
- **Recommended Methods**: `mpi4py`, third-party frameworks (e.g., Dask, Ray).
- **Reason**: These methods scale across multiple machines, making them suitable for large-scale computations that require distributed processing.

### Considerations and Best Practices

#### Avoid Sharing Mutable State
- **Multiprocessing**: Each process has its own memory space. Use shared memory or pipes for communication between processes.
- **Multithreading**: Ensure thread safety by using locks or other synchronization mechanisms to avoid race conditions.

#### Understand the Impact of GIL
- **CPU-bound tasks**: Multithreading may not improve performance due to the Global Interpreter Lock (GIL).
- **I/O-bound tasks**: Multithreading remains effective as the GIL has minimal impact on I/O operations.

#### Leverage High-level Parallel Tools
- **`concurrent.futures`**: Simplifies the management of threads and processes.
- **Third-party libraries**: Use libraries like `joblib` and `dask` to easily implement parallel computing.

#### Performance Testing and Optimization
- Use the `time` or `timeit` modules to measure execution time.
- Avoid excessive parallelization, as too many threads or processes may increase the overhead of context switching.

### Summary

Python provides various methods for parallel and concurrent programming, suitable for different scenarios.

#### Main Methods:
- **Multithreading (`threading`)**: Best for I/O-bound tasks.
- **Multiprocessing (`multiprocessing`)**: Suitable for CPU-bound tasks.
- **Coroutines (`asyncio`)**: Ideal for concurrent execution of many I/O operations.
- **MPI (`mpi4py`)**: Enables distributed memory parallel programming.

#### Choosing the Right Method:
- Select the appropriate parallel method based on the task's nature (CPU-bound or I/O-bound) and its scale.

#### Key Considerations:
- Understand the limitations and impact of the GIL.
- Pay attention to communication and synchronization between threads and processes.

## 7. Job Scheduling with Slurm

### Introduction to Slurm

#### What is Slurm?
- **Slurm** (Simple Linux Utility for Resource Management) is an open-source, scalable resource management and job scheduling system for high-performance computing (HPC) clusters.
- It manages the computing resources of HPC clusters, such as CPUs, memory, and accelerators (e.g., GPUs), and provides users with the ability to submit, monitor, and control jobs.

#### Why Use Slurm?
- **Resource Management**: Efficiently allocates and manages cluster resources.
- **Job Scheduling**: Provides advanced job queuing and scheduling strategies.
- **Scalability**: Supports various cluster sizes, from small clusters to supercomputers.
- **Open Source and Customizable**: Users can configure and extend Slurm according to their needs.

### Basic Concepts of Slurm

#### Key Components
- **Node**: A computer within the cluster.
- **Partition**: A logical grouping of nodes, used to define different resource limits and scheduling policies.
- **Job**: A computational task submitted by the user.
- **Job Step**: An individual process or set of processes executed as part of a job.

#### Job Types
- **Batch Job**: A non-interactive job submitted using a script.
- **Interactive Job**: A job where the user interacts directly with the job's processes, often used for tasks like interactive debugging via `srun`.


### Submitting Jobs

#### Submitting a Batch Job with `sbatch`
- **`sbatch`**: A command used to submit batch job scripts to Slurm.

##### Example Script:
`my_job_script.sh` or you can named it as `my_job_script.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=my_job          # Job name
#SBATCH --output=output_%j.txt     # Standard output and error output file (%j represents the job ID)
#SBATCH --ntasks=1                 # Total number of tasks (number of processes)
#SBATCH --time=01:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mem=1G                   # Requested memory size

echo "Hello, Slurm!"
```

```bash
sbatch my_job_script.sh
```
### Job Script Directives

- **`#SBATCH`**: Used to specify job resources and configurations in the job script. These directives should be placed at the top of the script.

#### Common `#SBATCH` Directives:
- `--job-name`: Sets the name of the job.
- `--output`: Specifies the file for standard output and error output.
- `--ntasks`: Specifies the total number of tasks, i.e., the number of processes.
- `--cpus-per-task`: Specifies the number of CPU cores used per task.
- `--nodes`: Specifies the number of nodes required.
- `--time`: Sets the maximum runtime for the job.
- `--mem`: Specifies the amount of memory requested.
- `--partition`: Specifies the partition to which the job is submitted.

### Resource Allocation

#### CPU Resources
- **`--ntasks`**: Specifies the total number of tasks, typically corresponding to the number of MPI processes.
- **`--cpus-per-task`**: Specifies the number of CPU cores per task, often used to define the number of OpenMP threads per process.

#### Example: Hybrid MPI and OpenMP Job

```bash
#!/bin/bash
#SBATCH --job-name=hybrid_job      # Job name
#SBATCH --output=output_%j.txt     # Standard output and error output file (%j represents the job ID)
#SBATCH --ntasks=4                 # Total number of MPI tasks (processes)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task (OpenMP threads)
#SBATCH --time=02:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mem=8G                   # Memory size requested

# Load necessary modules (e.g., MPI and OpenMP)
module load mpi
module load openmp

# Run the hybrid MPI/OpenMP program
mpirun -np 4 ./my_hybrid_mpi_openmp_program
```

### Memory Resources

- **`--mem`**: Specifies the total amount of memory required per node.
- **`--mem-per-cpu`**: Specifies the amount of memory allocated per CPU core.

### 7.4.3 GPU Resources

- **`--gres`**: Specifies generic resources, such as GPUs.

#### Example: GPU Resource Allocation

```bash
#!/bin/bash
#SBATCH --job-name=gpu_job           # Job name
#SBATCH --output=output_%j.txt       # Standard output and error output file (%j represents the job ID)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Total memory per node
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=01:00:00              # Maximum runtime (HH:MM:SS)

# Load necessary modules (e.g., CUDA or GPU libraries)
module load cuda

# Run the GPU-enabled program
./my_gpu_program
```

### Job Arrays

#### Purpose:
- Job arrays are used when you need to submit a large number of similar jobs, making it easier to manage and submit multiple jobs at once.

#### Submitting a Job Array

##### Example: Submitting 10 Jobs
```bash
#!/bin/bash
#SBATCH --job-name=array_job        # Job name
#SBATCH --output=array_output_%A_%a.txt  # Output file (%A for array job ID, %a for array task ID)
#SBATCH --array=1-10                # Submitting 10 jobs in the array

# Use SLURM_ARRAY_TASK_ID to get the index of the current job
echo "This is task number: $SLURM_ARRAY_TASK_ID"
```

### Controlling the Number of Concurrent Jobs

#### Limiting the Number of Concurrent Jobs

To limit the number of jobs running concurrently in a job array, you can specify a percentage value when submitting the job array.

##### Example: Limit to 10 Concurrent Jobs

```bash
#!/bin/bash
#SBATCH --job-name=array_job         # Job name
#SBATCH --output=array_output_%A_%a.txt  # Output file (%A for array job ID, %a for array task ID)
#SBATCH --array=1-100%10             # Submit 100 jobs, but limit to 10 concurrent jobs

# Use SLURM_ARRAY_TASK_ID to get the index of the current job
echo "This is task number: $SLURM_ARRAY_TASK_ID"
```

### Job Monitoring and Management

#### Viewing Job Status
- **`squeue`**: Use `squeue` to view the status of currently queued and running jobs.

```bash
squeue -u your_username
```

#### Cancelling a Job
**`scancel`**: Use `scancel` to cancel a job.

```bash
scancel job_id
```

### Viewing Job Details
**`sacct`**: Use `sacct` to view resource usage details for completed jobs.

```bash
sacct -j job_id --format=JobID,JobName,Partition,State,AllocCPUS,TotalCPU,Elapsed
```

### Example: Submitting an MPI Job

#### Job Script (mpi_job_script.sh / mpi_job_script.slurm)
```bash
#!/bin/bash
#SBATCH --job-name=mpi_job         # Job name
#SBATCH --output=mpi_output_%j.txt # Output file (%j represents the job ID)
#SBATCH --ntasks=4                 # Total number of tasks (processes)
#SBATCH --time=00:30:00            # Maximum runtime (HH:MM:SS)

module load mpi   # Load the MPI module (command may vary depending on the cluster configuration)

mpirun -np $SLURM_NTASKS ./your_mpi_program   # Run the MPI program using the number of tasks allocated by Slurm
```
#### Explanation:
- **`module load mpi`**: Loads the MPI environment. The specific command may vary depending on the cluster configuration.
- **`mpirun -np $SLURM_NTASKS`**: Runs the MPI program using the number of tasks allocated by Slurm (`$SLURM_NTASKS`), which corresponds to the number of processes specified in the job script.

### Example: Submitting an OpenMP Job

#### Job Script

```bash
#!/bin/bash
#SBATCH --job-name=openmp_job           # Job name
#SBATCH --output=openmp_output_%j.txt   # Output file (%j represents the job ID)
#SBATCH --ntasks=1                      # Total number of tasks (usually 1 for OpenMP jobs)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task (threads)
#SBATCH --time=00:30:00                 # Maximum runtime (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Set the number of threads for OpenMP

./your_openmp_program  # Run your OpenMP program
```
#### Explanation:
- **`export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`**: Sets the number of threads that OpenMP will use, based on the number of CPU cores allocated by Slurm (`$SLURM_CPUS_PER_TASK`). This ensures that OpenMP programs use the correct number of threads as specified in the job configuration.

### Example: Submitting a Python Job

#### Job Script

```bash
#!/bin/bash
#SBATCH --job-name=python_job          # Job name
#SBATCH --output=python_output_%j.txt  # Output file (%j represents the job ID)
#SBATCH --ntasks=1                     # Total number of tasks
#SBATCH --time=00:30:00                # Maximum runtime (HH:MM:SS)

module load anaconda  # Load Anaconda or Python environment

python your_script.py  # Execute the Python script
```

### Advanced Topics: Resource Reservation and Job Dependencies

#### Resource Reservation
- **Resource Reservation**: You can use the `salloc` command to reserve resources for interactive debugging.

```bash
salloc --ntasks=4 --time=01:00:00
```

### Job Dependencies

#### Purpose:
Job dependencies allow you to set relationships between jobs, so that a job starts only after another job has completed.

#### Example: Job Dependency

```bash
# Submit the first job
jobid1=$(sbatch --parsable job_script1.sh)

# Submit the second job, which depends on the first job completing successfully
sbatch --dependency=afterok:$jobid1 job_script2.sh
```
#### Explanation:
- **`--parsable`**: Ensures that `sbatch` only outputs the job ID, making it easy to capture and use in a script.
- **`--dependency`**: Sets the job dependency, defining when the job should start based on the status of another job.
- **`afterok`**: Specifies that the dependent job will only start if the specified job completes successfully.

### Common Issues and Debugging Techniques
#### Job Fails to Start

##### Possible Causes:
- Insufficient resources, causing a long wait in the queue.
- Unreasonable resource requests in the job script.

#### Solutions:
- Review and adjust resource requests in the job script.
- Contact the system administrator to check the cluster's status.

#### Job Crashes or Exits Abnormally

##### Possible Causes:
- Program error or crash.
- Insufficient memory or timeout.

##### Solutions:
- Check the output and error files for details to pinpoint the issue.
- Increase memory or adjust the time limit.

#### Environment or Module Not Found

##### Possible Causes:
- The environment module is not correctly loaded in the job script.
- Environment variables are not set.

##### Solutions:
- Explicitly load the required modules in the job script.
- Ensure the environment setup in the job script matches the interactive environment.

### Summary

Slurm is a commonly used resource management and job scheduling system in HPC clusters, offering a rich set of features to manage computing resources and jobs.

#### Key Features:
- **Job Submission**: Use `sbatch` to submit batch jobs.
- **Resource Allocation**: Specify resource requirements using `#SBATCH` directives.
- **Job Management**: Monitor and control jobs using commands like `squeue` and `scancel`.
- **Job Arrays**: Easily submit large numbers of similar jobs using job arrays.

#### Best Practices:
- **Request Resources Appropriately**: Request CPU, memory, and time resources based on program needs to avoid wasting resources or falling short.
- **Use Module to Manage Environments**: Explicitly load necessary environment modules in the job script to ensure consistency between the job environment and the development environment.
- **Debug and Test**: Before submitting large-scale jobs, perform small-scale tests to ensure the program runs correctly.

### 8. Distributed Training (Machine Learning)

#### Basic Concepts of Distributed Training

##### Why is Distributed Training Needed?
- **Large-scale datasets**: Datasets may be too large to fit into the memory of a single node.
- **Large models**: Deep learning models have an enormous number of parameters, making the training process computationally intensive.
- **Accelerating training**: Distributed training can shorten training times and accelerate model iteration.

##### Parallelization Strategies
- **Data Parallelism**: The dataset is divided into multiple parts, and model replicas are trained simultaneously on different nodes.
- **Model Parallelism**: Different parts of the model are distributed across different nodes for simultaneous computation.

### Data Parallelism

#### Principles of Data Parallelism
- **Model Replication**: Each computing node holds a complete replica of the model.
- **Data Sharding**: The training dataset is divided into multiple subsets, which are distributed to different nodes.
- **Gradient Aggregation**: Each node computes its own gradients, and these gradients are then aggregated (e.g., averaged) to update the model parameters.

#### Implementation of Data Parallelism
- **Synchronous Updates**: All nodes compute their gradients, and only after all nodes have finished, the parameters are updated.
- **Asynchronous Updates**: Nodes independently update parameters, which can lead to inconsistent parameters.

#### Advantages and Disadvantages
- **Advantages**:
  - **Easy to implement**: Most deep learning frameworks support data parallelism.
  - **Good scalability**: Can utilize more nodes to speed up training.
- **Disadvantages**:
  - **High communication overhead**: Gradient synchronization is required after each training step.
  - **Limited by model size**: Each node must store a full copy of the model.

### Model Parallelism

#### Principles of Model Parallelism
- **Model Partitioning**: Different parts of the model are distributed across different nodes.
- **Forward and Backward Propagation**: Activation values and gradients are passed between nodes during forward and backward propagation.

#### Implementation of Model Parallelism
- **Layer-wise Partitioning**: Different layers of the model are placed on different nodes.
- **Tensor Partitioning**: Weights or activations within the same layer are split across different nodes.

#### Advantages and Disadvantages
- **Advantages**:
  - **Suitable for large models**: Useful when the model is too large to fit on a single node.
- **Disadvantages**:
  - **Complex implementation**: Requires careful partitioning of the model and handling communication between nodes.
  - **Communication overhead**: Frequent communication between nodes can impact performance.

###  Hybrid Parallelism
- **Combining Data and Model Parallelism**: In some scenarios, both strategies can be used simultaneously to make the most of available resources.

### Distributed Training with TensorFlow

#### TensorFlow's Distributed Architecture
- **Strategy API**: The `tf.distribute` module provides various strategies to implement distributed training.

#### Data Parallel Strategy
- **`MirroredStrategy`**:
  - **Purpose**: Implements data parallelism on a single machine with multiple GPUs.

Example usage:

```python
import tensorflow as tf

# Create a MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope and build the model
with strategy.scope():
   model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model using data parallelism across multiple GPUs
model.fit(dataset, epochs=10)
```

or you can change the `strategy`  to `MultiWorkerMirroredStrategy` for multi-worker-multi-GPU:
- **Purpose**: Implements data parallelism across multiple machines with multiple GPUs.

#### Environment Setup
- **TF_CONFIG Environment Variable**: The `TF_CONFIG` environment variable must be set to specify the cluster configuration for distributed training.

Example of setting `TF_CONFIG` for a multi-worker setup:
```bash
export TF_CONFIG='{
  "cluster": {
    "worker": ["worker0.example.com:12345", "worker1.example.com:23456"]
  },
  "task": {"type": "worker", "index": 0}
}
```

```python
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
```

### 9.6 Distributed Training with PyTorch

#### 9.6.1 PyTorch's Distributed Package
- **`torch.distributed`**: PyTorch's native distributed training package, which supports multiple backends such as NCCL, Gloo, and MPI.

#### 9.6.2 Data Parallelism
- **`torch.nn.DataParallel`**:
  - **Purpose**: Implements data parallelism on a single machine with multiple GPUs.
  
Example usage:

```python
  import torch
  import torch.nn as nn
  import torch.optim as optim

  # Define a simple model
  model = nn.Sequential(
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 10)
  )

  # Wrap the model with DataParallel to enable multi-GPU training
  model = nn.DataParallel(model)

  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Move model to the available GPUs
  model.cuda()

  ... training ...
 ```

### `torch.nn.parallel.DistributedDataParallel`

- **Purpose**: Implements data parallelism across multiple machines with multiple GPUs, offering better performance than `DataParallel`.

#### Environment Setup
- To use `DistributedDataParallel`, you need to initialize the distributed environment.

#### Example Setup:
```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Create the dataset and data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Define the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).cuda()

    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
```

Initiate the training through (if there're 2 GPU):
```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### Key Components

- **`torch.distributed.launch`**: A tool provided by PyTorch to launch multiple processes for distributed training. It helps to set up the environment for each process, ensuring proper communication between nodes and GPUs.

- **`DistributedSampler`**: Ensures that each process (node) gets a different part of the dataset, preventing duplication of data across processes. This is important for training efficiency in distributed settings.

- **`DistributedDataParallel` (DDP)**: Enables data parallelism across multiple GPUs. DDP synchronizes gradients during the backward pass to ensure consistent parameter updates across GPUs, providing better performance compared to `DataParallel` in multi-GPU setups.

### Performance Optimization Techniques

#### Reducing Communication Overhead
- **Gradient Compression**: Compress or quantize gradients to reduce the amount of data communicated between nodes.
- **Overlapping Communication and Computation**: Perform communication in the background while computation continues in parallel, reducing idle time.

#### Adjusting Batch Size
- **Increase Batch Size**: In data parallelism, increasing the batch size can improve GPU utilization and training efficiency.
- **Learning Rate Adjustment**: As batch size increases, you may need to adjust the learning rate to maintain optimal training convergence.

#### Mixed Precision Training
- **Concept**: Use half-precision (FP16) for computations to reduce memory usage and computational load.
- **Implementation**: Deep learning frameworks typically provide support for mixed precision training, enabling efficient use of hardware resources like GPUs.

Example in PyTorch for Mixed Precision Training:
  ```python
  from torch.cuda.amp import autocast, GradScaler
  
  # Initialize GradScaler for mixed precision
  scaler = GradScaler()

  for inputs, labels in dataloader:
      inputs, labels = inputs.cuda(), labels.cuda()

      optimizer.zero_grad()

      # Perform forward pass and backward pass with mixed precision
      with autocast():
          outputs = model(inputs)
          loss = criterion(outputs, labels)

      # Scale gradients and perform optimizer step
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```

  ### Common Issues and Solutions

#### Parameter Update Inconsistency
- **Problem**: In asynchronous updates, model parameters may become unsynchronized across different nodes.
- **Solution**: Use synchronous updates or introduce a **Parameter Server** to coordinate parameter updates across nodes.

#### Low Resource Utilization
- **Problem**: High communication overhead leads to idle computational resources, reducing efficiency.
- **Solution**: Optimize communication, adjust the way the model is partitioned, and use an efficient communication backend (e.g., NCCL for multi-GPU setups).

#### Training Does Not Converge
- **Problem**: Changes in batch size can lead to instability in the training process, preventing convergence.
- **Solution**: Adjust the learning rate accordingly and consider using a **learning rate warmup** strategy to stabilize the training during the initial phase.
