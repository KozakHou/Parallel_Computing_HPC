#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <vector>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI environment

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes

    int n = 4; // Size of the matrix
    if (argc == 2) { // If the size of the matrix is provided as a command line argument
        n = atoi(argv[1]); // Convert the argument to int
    }

    int *A = nullptr; // Matrix A
    int *B = nullptr; // Matrix B
    int *C = nullptr; // Matrix C

    if (world_rank == 0) { // Process 0: Initialize the matrices
        A = new int[n * n];
        B = new int[n * n];
        C = new int[n * n];
        for (int i = 0; i < n * n; ++i) {
            A[i] = i + 1; // Initialize matrix A with values from 1 to n^2
            B[i] = i + 1; // Initialize matrix B with values from 1 to n^2
        }
    } else {
        B = new int[n * n]; // Other processes need B for computation
    }

    // Calculate the number of rows each process will handle
    int rows_per_proc = n / world_size;
    int remainder = n % world_size;

    // Adjust for processes that need to handle the extra rows
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_rows = sendcounts[world_rank] / n;
    int *local_A = new int[sendcounts[world_rank]]; // Local matrix A for each process
    int *local_C = new int[sendcounts[world_rank]]; // Local matrix C for each process

    MPI_Scatterv(A, sendcounts.data(), displs.data(), MPI_INT, local_A, sendcounts[world_rank], MPI_INT, 0, MPI_COMM_WORLD); // Scatter the matrix A to all processes

    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the matrix B to all processes

    // Perform the matrix multiplication
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            local_C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    MPI_Gatherv(local_C, sendcounts[world_rank], MPI_INT, C, sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD); // Gather the results from all processes

    if (world_rank == 0) {
        std::cout << "Matrix C:" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << C[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Clean up
    if (world_rank == 0) {
        delete[] A;
        delete[] B;
        delete[] C;
    } else {
        delete[] B;
    }
    delete[] local_A;
    delete[] local_C;

    MPI_Finalize(); // Finalize the MPI environment
    return 0;
}
