#include<iostream>
#include<mpi.h>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv); // Initialize MPI enviroment

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 

    int data = -1;
    MPI_Request request;
    if (world_rank == 0){ // Process 0: Send data to process 1
        data = 100;
        MPI_Isend(&data, 1 , MPI_INT, 1, 0, MPI_COMM_WORLD, &request); // Non-blocking send
        // The arguments are: &data, count, datatype, destination, tag, comm, request
        MPI_Wait(&request, MPI_STATUS_IGNORE); // Wait for the request to complete
        std::cout << "Process 0 completed non-blocking send" << std::endl;
    } else if (world_rank == 1){ // Process 1: Receive data from process 0
        MPI_Irecv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request); // Non-blocking receive
        // The arguments are: &data, count, datatype, source, tag, comm, request
        MPI_Wait(&request, MPI_STATUS_IGNORE); // Wait for the request to complete
        std::cout << "Process 1 completed non-blocking receive" << std::endl;
    }
    
    MPI_Finalize(); // Finalize MPI environment
    return 0;
}
