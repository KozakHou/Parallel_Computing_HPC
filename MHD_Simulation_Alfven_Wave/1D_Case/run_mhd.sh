#!/bin/bash
#SBATCH --job-name=mhd_simulation
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=4            # number of MPI tasks
#SBATCH --cpus-per-task=4     # number of OpenMP threads per MPI task
#SBATCH --time=01:00:00       # maximum walltime
#SBATCH --partition=compute   # partition

# load modules
module load mpi
module load gcc

# Set the number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile the code
mpic++ -fopenmp main.cpp mhd.cpp -o mhd_simulation

# run 
mpirun -np $SLURM_NTASKS ./mhd_simulation
