#!/bin/bash
#SBATCH --job-name=GPU_Slurm
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1 # 1 task
#SBATCH --cpus-per-task=4  # 4 cores per task
#SBATCH --mem=16G # 16GB of memory
#SPBATCH --gres=gpu:1 # 1 GPU
#SBATCH --time=01:00:00 # 1 hour

module load cuda
./my_gpu_program