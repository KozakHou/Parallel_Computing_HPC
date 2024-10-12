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
