#!/bin/bash
#SBATCH --job-name=mpi_job         # Job name
#SBATCH --output=mpi_output_%j.txt # Output file (%j represents the job ID)
#SBATCH --ntasks=4                 # Total number of tasks (processes)
#SBATCH --time=00:30:00            # Maximum runtime (HH:MM:SS)

module load mpi   # Load the MPI module (command may vary depending on the cluster configuration)

mpirun -np $SLURM_NTASKS ./your_mpi_program   # Run the MPI program using the number of tasks allocated by Slurm