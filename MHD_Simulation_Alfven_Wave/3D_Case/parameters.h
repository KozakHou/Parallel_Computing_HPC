// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// physical parameters
const double rho0 = 1.0;     // density
const double p0 = 1.0;       // pressure
const double B0 = 1.0;       // magnetic field strength
const double adiabatic_index = 5.0/3.0;    // adiabatic index
const double mu0 = 1.0;      // magnetic permeability, set to 1 for simplicity

// simulation parameters
const int NX = 64;           // number of grid points in X direction
const int NY = 64;           // number of grid points in Y direction
const int NZ = 64;           // number of grid points in Z direction
const double x_min = 0.0;    // X direction range start
const double x_max = 1.0;    // X direction range end  
const double y_min = 0.0;    // Y direction range start
const double y_max = 1.0;    // Y direction range end
const double z_min = 0.0;    // Z direction range start
const double z_max = 1.0;    // Z direction range end
const double dx = (x_max - x_min) / NX; // X direction space step
const double dy = (y_max - y_min) / NY; // Y direction space step
const double dz = (z_max - z_min) / NZ; // Z direction space step
const double CFL = 0.5;      // CFL number
const double t_max = 1.0;    // simulation total time

// output parameters
const int output_interval = 10; // output interval

#endif
