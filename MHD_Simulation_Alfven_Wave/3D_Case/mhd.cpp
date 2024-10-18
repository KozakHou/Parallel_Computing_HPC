// mhd.cpp
#include "mhd.h"
#include "parameters.h"
#include <fstream>
#include <omp.h>
#include <cmath>


MHD::MHD(int nx_local, int ny, int nz) : nx_local(nx_local), ny(ny), nz(nz) {
    rho.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    u.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    v.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    w.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    p.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    Bx.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    By.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));
    Bz.resize(nx_local, std::vector<std::vector<double>>(ny, std::vector<double>(nz)));

    rho_new = rho;
    u_new = u;
    v_new = v;
    w_new = w;
    p_new = p;
    Bx_new = Bx;
    By_new = By;
    Bz_new = Bz;
}

void MHD::initialize(int x_offset) {
    // initialize state, set initial conditions for three-dimensional Alfven wave
    for (int i = 0; i < nx_local; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                double x = x_min + (i + x_offset) * dx;
                double y = y_min + j * dy;
                double z = z_min + k * dz;
                rho[i][j][k] = rho0;
                u[i][j][k] = 0.1 * sin(2 * M_PI * x);
                v[i][j][k] = 0.1 * sin(2 * M_PI * y);
                w[i][j][k] = 0.1 * sin(2 * M_PI * z);
                p[i][j][k] = p0;
                Bx[i][j][k] = B0;
                By[i][j][k] = B0;
                Bz[i][j][k] = B0;
            }
        }
    }
}

void MHD::step(double dt) {
    // FEM: finite difference method
    // here only give simplified update equations, in actual application, more complex processing is needed
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < nx_local - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int k = 1; k < nz - 1; ++k) {
                // mass conservation equation
                rho_new[i][j][k] = rho[i][j][k] - dt * (
                    ( (rho[i][j][k] * u[i][j][k] - rho[i-1][j][k] * u[i-1][j][k]) / dx ) +
                    ( (rho[i][j][k] * v[i][j][k] - rho[i][j-1][k] * v[i][j-1][k]) / dy ) +
                    ( (rho[i][j][k] * w[i][j][k] - rho[i][j][k-1] * w[i][j][k-1]) / dz )
                );
                // momentum conservation equation
                u_new[i][j][k] = u[i][j][k] - dt * (
                    ( (u[i][j][k]*u[i][j][k] + p[i][j][k]/rho[i][j][k]) - (u[i-1][j][k]*u[i-1][j][k] + p[i-1][j][k]/rho[i-1][j][k]) ) / dx
                );
                
                v_new[i][j][k] = v[i][j][k] - dt * (
                    ( (v[i][j][k]*v[i][j][k] + p[i][j][k]/rho[i][j][k]) - (v[i][j-1][k]*v[i][j-1][k] + p[i][j-1][k]/rho[i][j-1][k]) ) / dy
                );

                w_new[i][j][k] = w[i][j][k] - dt * (
                    ( (w[i][j][k]*w[i][j][k] + p[i][j][k]/rho[i][j][k]) - (w[i][j][k-1]*w[i][j][k-1] + p[i][j][k-1]/rho[i][j][k-1]) ) / dz
                );

                // energy conservation equation
                p_new[i][j][k] = p[i][j][k] - dt * (
                    (u[i][j][k]*u[i][j][k] + v[i][j][k]*v[i][j][k] + w[i][j][k]*w[i][j][k]) * (adiabatic_index - 1) +
                    ((p[i][j][k] - p[i-1][j][k]) / dx) +
                    ((p[i][j][k] - p[i][j-1][k]) / dy) +
                    ((p[i][j][k] - p[i][j][k-1]) / dz)
                );

                // magnetic field conservation equation
                Bx_new[i][j][k] = Bx[i][j][k] - dt * (
                    ( (Bx[i][j][k] * u[i][j][k] - Bx[i-1][j][k] * u[i-1][j][k]) / dx ) +
                    ( (Bx[i][j][k] * v[i][j][k] - Bx[i][j-1][k] * v[i][j-1][k]) / dy ) +
                    ( (Bx[i][j][k] * w[i][j][k] - Bx[i][j][k-1] * w[i][j][k-1]) / dz )
                );

                By_new[i][j][k] = By[i][j][k] - dt * (
                    ( (By[i][j][k] * u[i][j][k] - By[i-1][j][k] * u[i-1][j][k]) / dx ) +
                    ( (By[i][j][k] * v[i][j][k] - By[i][j-1][k] * v[i][j-1][k]) / dy ) +
                    ( (By[i][j][k] * w[i][j][k] - By[i][j][k-1] * w[i][j][k-1]) / dz )
                );

                Bz_new[i][j][k] = Bz[i][j][k] - dt * (
                    ( (Bz[i][j][k] * u[i][j][k] - Bz[i-1][j][k] * u[i-1][j][k]) / dx ) +
                    ( (Bz[i][j][k] * v[i][j][k] - Bz[i][j-1][k] * v[i][j-1][k]) / dy ) +
                    ( (Bz[i][j][k] * w[i][j][k] - Bz[i][j][k-1] * w[i][j][k-1]) / dz )
                );
            }
        }
    }

    // boundary conditions (here simply processed, in actual application, more rigorous processing is needed)
    // for example, periodic boundary conditions
    // update state
    rho.swap(rho_new);
    u.swap(u_new);
    v.swap(v_new);
    w.swap(w_new);
    p.swap(p_new);
    Bx.swap(Bx_new);
    By.swap(By_new);
    Bz.swap(Bz_new);
}

void MHD::save_data(int timestep, int x_offset) {
    // save data to file
    std::ofstream file("Results/output_" + std::to_string(timestep) + "_" + std::to_string(x_offset) + ".dat");
    for (int i = 0; i < nx_local; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                double x = x_min + (i + x_offset) * dx;
                double y = y_min + j * dy;
                double z = z_min + k * dz;
                file << x << " " << y << " " << z << " "
                     << rho[i][j][k] << " "
                     << u[i][j][k] << " "
                     << v[i][j][k] << " "
                     << w[i][j][k] << " "
                     << p[i][j][k] << " "
                     << Bx[i][j][k] << " "
                     << By[i][j][k] << " "
                     << Bz[i][j][k] << "\n";
            }
        }
    }
    file.close();
}
