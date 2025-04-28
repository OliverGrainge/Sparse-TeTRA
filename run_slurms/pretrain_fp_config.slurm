#!/bin/bash

#SBATCH --job-name=fp_config    # Job name
#SBATCH --partition=swarm_a100
#SBATCH --nodes=1                  # Run on one node
#SBATCH --cpus-per-task=8         # Number of CPU cores per task (matching num_workers)
#SBATCH --time=72:00:00           # Time limit hrs:min:sec
#SBATCH --output=logs_slurm/pretrain_%j.log   # Standard output log
#SBATCH --error=logs_slurm/pretrain_%j.err    # Standard error log
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --ntasks-per-node=2

# Activate conda environment 
export WANDB_MODE=offline

# Run the training script
srun python pretrain.py --config run_configs/pretrain/fp_config.yaml
