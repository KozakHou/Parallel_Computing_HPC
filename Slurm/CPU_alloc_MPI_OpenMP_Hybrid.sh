#!/bin/bash
#SBATCH --job-name=CPU_alloc_MPI_OpenMP_Hybrid
#SBATCH --output=output_%j.txt # output file (%j is replaced by the job ID)
#SBATCH --ntasks=4 # 4 tasks
#SBATCH --cpus-per-task= 8 # 8 cores per task
#SBATCH --time=02:00:00 # 2 hours
#SBATCH --mem=8G # 8GB of memory

module load openmpi
module load mpi
mpirun -np 4 ./my_mpi_program

