// mhd.cpp
#include "mhd.h"
#include "parameters.h"
#include <fstream>
#include <omp.h>
#include <cmath>
#include <filesystem>

MHD::MHD(int nx) : nx(nx) {
    rho.resize(nx);
    u.resize(nx);
    p.resize(nx);
    B.resize(nx);

    rho_new.resize(nx);
    u_new.resize(nx);
    p_new.resize(nx);
    B_new.resize(nx);
}

void MHD::initialize() {
    // Initialize for the Alfven wave state
    for (int i = 0; i < nx; ++i) {
        double x = x_min + i * dx;
        rho[i] = rho0;
        u[i] = 0.1 * sin(2 * M_PI * x);
        p[i] = p0;
        B[i] = B0;
    }
}

void MHD::step(double dt) {
    // Finite Difference Method
    #pragma omp parallel for
    for (int i = 1; i < nx - 1; ++i) {
        // Simplified MHD equations
        rho_new[i] = rho[i] - dt / dx * (rho[i] * u[i] - rho[i - 1] * u[i - 1]);
        u_new[i] = u[i] - dt / dx * ((u[i] * u[i] + p[i] / rho[i]) - (u[i - 1] * u[i - 1] + p[i - 1] / rho[i - 1]));
        p_new[i] = p[i] - dt / dx * (u[i] * p[i] - u[i - 1] * p[i - 1]);
        B_new[i] = B[i]; // Assume the magnetic field is constant
    }

    // Bondary Conditions: Periodic
    rho_new[0] = rho_new[1];
    rho_new[nx - 1] = rho_new[nx - 2];

    u_new[0] = u_new[1];
    u_new[nx - 1] = u_new[nx - 2];

    p_new[0] = p_new[1];
    p_new[nx - 1] = p_new[nx - 2];

    B_new[0] = B_new[1];
    B_new[nx - 1] = B_new[nx - 2];

    // Update the state
    rho.swap(rho_new);
    u.swap(u_new);
    p.swap(p_new);
    B.swap(B_new);
}

void MHD::save_data(int timestep) {
    // Save the data to a file
    std::filesystem::create_directories("Result");
    std::ofstream file("Result/output_" + std::to_string(timestep) + ".dat");
    for (int i = 0; i < nx; ++i) {
        double x = x_min + i * dx;
        file << x << " " << rho[i] << " " << u[i] << " " << p[i] << " " << B[i] << "\n";
    }
    file.close();
}

