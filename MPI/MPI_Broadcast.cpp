#include<iostream>
#include<mpi.h>

int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process

    int data; // Data to be broadcasted
    if (world_rank == 0){
        data = 100; 
    }

    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the data to all processes
    std::cout << "Process " << world_rank << " received data " << data << std::endl; // Print the received data

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}