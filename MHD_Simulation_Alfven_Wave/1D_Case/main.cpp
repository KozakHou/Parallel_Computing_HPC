// main.cpp
#include <mpi.h>
#include "mhd.h"
#include "parameters.h"
#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Calculate the grid size for each process
    int nx_local = NX / world_size;
    int start = world_rank * nx_local;
    int end = (world_rank == world_size - 1) ? NX : start + nx_local;

    MHD mhd(nx_local);
    mhd.initialize();

    double t = 0.0;
    int timestep = 0;

    while (t < t_max) {
        // Time step
        double dt = CFL * dx / (std::abs(1.0)); // Assume the wave speed is 1.0

        mhd.step(dt);

        t += dt;
        timestep++;

        // Output the result every output_interval steps
        if (timestep % output_interval == 0) {
            // Gather the data to the root process
            // Note: The root process is responsible for saving the data
            std::vector<double> rho_global(NX), u_global(NX), p_global(NX), B_global(NX);

            MPI_Gather(mhd.rho.data(), nx_local, MPI_DOUBLE, rho_global.data(), nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(mhd.u.data(), nx_local, MPI_DOUBLE, u_global.data(), nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(mhd.p.data(), nx_local, MPI_DOUBLE, p_global.data(), nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(mhd.B.data(), nx_local, MPI_DOUBLE, B_global.data(), nx_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (world_rank == 0) {
                // Save the data
                std::ofstream file("Result/output_" + std::to_string(timestep) + ".dat");
                for (int i = 0; i < NX; ++i) {
                    double x = x_min + i * dx;
                    file << x << " " << rho_global[i] << " " << u_global[i] << " " << p_global[i] << " " << B_global[i] << "\n";
                }
                file.close();
                std::cout << "Time: " << t << ", Output saved at timestep " << timestep << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
