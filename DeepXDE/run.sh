#!/bin/bash
#SBATCH --job-name=python         # Job name
#SBATCH --output=log_slurm.%j.out # Output file
#SBATCH --error=log_slurm.%j.err  # Error file
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --partition=gpu           # Partition/queue name
#SBATCH --time=1:00:00            # time limit

# Activate the conda environment
. ~/.bashrc
module load cudnn8.4-cuda11.4
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export DDE_BACKEND=tensorflow
export TF_GPU_ALLOCATOR=cuda_malloc_async
conda activate tf-gpu

# Run the Python script
python3 PINN.py