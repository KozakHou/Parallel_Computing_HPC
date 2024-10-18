// main.cpp
#include <mpi.h>
#include "mhd.h"
#include "parameters.h"
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char *argv[]) {
    // initialize MPI environment
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // calculate the number of X-direction grid points each process needs to handle
    int nx_local = NX / world_size;
    int remainder = NX % world_size;
    if (world_rank < remainder) {
        nx_local += 1;
    }
    int x_offset = (NX / world_size) * world_rank + std::min(world_rank, remainder);

    MHD mhd(nx_local, NY, NZ);
    mhd.initialize(x_offset);

    double t = 0.0;
    int timestep = 0;
    double max_velocity = 5.0;

    while (t < t_max) {
        // calculate time step, simplified processing
        double dt = CFL * std::min(std::min(dx, dy), dz) / max_velocity; // assume maximum speed is 1.0

        mhd.step(dt);

        t += dt;
        timestep++;

        // output results every few steps
        if (timestep % output_interval == 0) {
            // save local data for each process
            mhd.save_data(timestep, x_offset);
            if (world_rank == 0) {
                std::cout << "Time: " << t << ", Output saved at timestep " << timestep << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
