// mhd.h
#ifndef MHD_H
#define MHD_H

#include <vector>

class MHD {
public:
    MHD(int nx);
    void initialize();
    void step(double dt);
    void save_data(int timestep);

    std::vector<double> rho, u, p, B;

private:
    int nx;
    std::vector<double> rho_new, u_new, p_new, B_new;
};

#endif
