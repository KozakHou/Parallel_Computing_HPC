// parameters.h
#ifndef PARAMETERS_H
#define PARAMETERS_H

// physical constants
const double rho0 = 1.0;     // initial density
const double p0 = 1.0;       // initial pressure
const double B0 = 1.0;       // initial magnetic field
const double adiabatic_index = 1.4;    // adiabatic index

// simulation parameters
const int NX = 1000;         // spatial grid number
const double x_min = 0.0;    // spatial range start
const double x_max = 1.0;    // spatial range end
const double dx = (x_max - x_min) / NX; // step size

const double CFL = 0.5;      // CFL number
const double t_max = 1.0;    // simulation time

// output parameters
const int output_interval = 10; // output interval

#endif
