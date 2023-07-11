#!/bin/bash
#SBATCH -J python           # job name
#SBATCH -o log_slurm.o%j    # output and error file name (%j expands to jobID)
#SBATCH -n 1                # total number of tasks requested
#SBATCH -N 1                # number of nodes you want to run on
#SBATCH -p bsudfq           # queue (partition) for R2 use defq
#SBATCH -t 1:00:00         # run time (hh:mm:ss) - 12.0 hours in this example.

# Activate the conda environment
# Replace environmentName with your environment name

. ~/.bashrc

cd .. && source ./bin/activate && cd DeepXDE

conda activate climate

python PINN.py