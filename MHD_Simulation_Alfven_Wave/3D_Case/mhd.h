// mhd.h
#ifndef MHD_H
#define MHD_H

#include <vector>

class MHD {
public:
    MHD(int nx_local, int ny, int nz);
    void initialize(int x_offset);
    void step(double dt);
    void save_data(int timestep, int x_offset);

    // state variables
    std::vector<std::vector<std::vector<double>>> rho, u, v, w, p, Bx, By, Bz;

private:
    int nx_local, ny, nz;
    // temporary variables
    std::vector<std::vector<std::vector<double>>> rho_new, u_new, v_new, w_new, p_new, Bx_new, By_new, Bz_new;
};

#endif