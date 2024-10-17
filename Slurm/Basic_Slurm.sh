#!/bin/bash
#SBATCH --job-name=Basic_Slurm
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mpm=16gb

echo "Hello World"