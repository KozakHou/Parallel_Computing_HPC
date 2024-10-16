#include<iostream>
#include<mpi.h>
#include<cstdlib>
#include<ctime>

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv); // Initialize MPI environment

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes

    long long int num_tosses = 1000000; // Number of tosses
    if (argc == 2){  // If the number of tosses is provided as a command line argument
        num_tosses = atoll(argv[1]); // Convert the argument to long long int
    }

    long long int local_tosses = num_tosses / world_size; // Number of tosses per process
    long long int local_count = 0;

    unsigned int seed = time(NULL) * world_rank; // Seed for the random number generator

    for (long long int i = 0; i < local_tosses; ++i){ // Perform the Monte Carlo simulation
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0; // Generate a random number between -1 and 1
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0; // Generate a random number between -1 and 1
        if (x * x + y * y <= 1.0){ // Check if the point is inside the circle
            local_count++; // Increment the count
        }
    }

    long long int global_count;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce the local counts to the global count
    // The arguments are: sendbuf, recvbuf, count, datatype, op, root, comm

    if (world_rank == 0){
        double pi_estimate = 4.0 * global_count / num_tosses; // Estimate the value of pi
        std::cout << "Estimated Pi = " << pi_estimate << std::endl; // Print the estimated value of pi
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}

// after compiling the code, run the following command:
// mpirun -n 4 ./MPI_Pi_Appr or mpirun -n 4 ./MPI_Pi_Appr 10000000 (to specify the number of tosses)