#!/bin/bash
#SBATCH --job-name=Control_Job_Array
#SBATCH --output=array_output_%j.txt
#SBATCH --array=1-100%10 # submit 100 tasks, 10 at a time (concurrently)
#SBATCH --time=00:10:00

echo "This is task number $SLURM_ARRAY_TASK_ID"