#include<iostream>
#include<mpi.h>
#include<cstdlib>
#include<ctime>

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv); // Initialize MPI environment

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes

    long long int num_elements = 1000000; // Number of elements
    if (argc == 2){  // If the number of elements is provided as a command line argument
        num_elements = atoll(argv[1]); // Convert the argument to long long int
    }

    long long int local_elements = num_elements / world_size; // Number of elements per process
    long long int local_sum = 0;

    int *data = new int[num_elements]; // Array of integers
    if (world_rank == 0){ // Process 0: Initialize the array
        for (long long int i = 0; i < num_elements; ++i){
            data[i] = i + 1; // Initialize the array with values from 1 to num_elements
        }
    }

    int *local_data = new int[local_elements]; // Local array for each process
    MPI_Scatter(data, local_elements, MPI_INT, local_data, local_elements, MPI_INT, 0, MPI_COMM_WORLD); // Scatter the data to all processes
    // The arguments are: sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm

    for (long long int i = 0; i < local_elements; ++i){ // Calculate the local sum
        local_sum += local_data[i];
    }

    long long int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce the local sums to the global sum
    // The arguments are: sendbuf, recvbuf, count, datatype, op, root, comm

    if (world_rank == 0){
        std::cout << "Sum of elements from 1 to " << num_elements << " = " << global_sum << std::endl; // Print the global sum
    }

    delete[] data;
    delete[] local_data;

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}