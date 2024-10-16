#include<iostream>
#include<mpi.h>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv); // Initialize MPI enviroment

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process

    int local_sum = world_rank; // local sum of the process
    int global_sum; 

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce the local sum to the global sum
    // The arguments are: &send_data, &recv_data, count, datatype, operation, root, comm

    if (world_rank == 0){ // if the process is the root process
        std::cout << "Total sum = " << global_sum << std::endl; 
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}