#!/bin/bash
#SBATCH --job-name=python_job          # Job name
#SBATCH --output=python_output_%j.txt  # Output file (%j represents the job ID)
#SBATCH --ntasks=1                     # Total number of tasks
#SBATCH --time=00:30:00                # Maximum runtime (HH:MM:SS)

module load anaconda  # Load Anaconda or Python environment

python your_script.py  # Execute the Python script
