#include<iostream>
#include<mpi.h>

int main(int argc, char *argv[]){ // argc: number of arguments, argv: arguments
    
    MPI_Init(&argc, &argv); // Initialize MPI environment

    int world_size; // Number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes

    int world_rank; // Rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process

    char processor_name[MPI_MAX_PROCESSOR_NAME]; // Name of the processor
    int name_len; // length of the processor name
    MPI_Get_processor_name(processor_name, &name_len); // Get the processor name

    std::cout << "Hello from processor " << processor_name << ", rank "
                << world_rank << " out of " << world_size << " processors" << std::endl;

    MPI_Finalize(); // Finalize MPI environment

    return 0;

}

// Compile: mpic++ Basic_Structure.cpp -o Basic_Structure
// Run: mpirun -np 4 ./Basic_Structure (for 4 processes, np: number of processes)