#!/bin/bash
#SBATCH --job-name=openmp_job           # Job name
#SBATCH --output=openmp_output_%j.txt   # Output file (%j represents the job ID)
#SBATCH --ntasks=1                      # Total number of tasks (usually 1 for OpenMP jobs)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task (threads)
#SBATCH --time=00:30:00                 # Maximum runtime (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Set the number of threads for OpenMP

./your_openmp_program  # Run your OpenMP program