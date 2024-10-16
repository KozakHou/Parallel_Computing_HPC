#include<iostream>
#include<mpi.h>


int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv); // Initialize MPI environment (& is used to pass the address of the variable)

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the number of processes

    int number;
    if (world_rank == 0){
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); 
        std::cout << "Process 0 sent number " << number << " to process 1" << std::endl;
    } else if (world_rank == 1){
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process 1 received number " << number << " from process 0" << std::endl;
    }

    MPI_Finalize();
    return 0;
    
}