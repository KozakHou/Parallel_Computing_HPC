#!/bin/bash
#SBATCH --job-name=array_job # Job name
#SBATCH --output=output_%j.txt # output file (%j is replaced by the job ID)
#SBATCH --array=1-10 # Array of tasks

echo "Task ID is $SLURM_ARRAY_TASK_ID"