#!/bin/bash
#SBATCH --job-name=hopper_training_64_1     # Job name
#SBATCH --partition=gpunodes          # Partition to use (adjust if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory allocation
#SBATCH --time=12:00:00               # Maximum runtime (12 hour)
#SBATCH --output=hopper_training_64_1.out  # Standard output file
#SBATCH --error=hopper_training_64_1.err   # Standard error file

# Activate your Conda environment
# conda activate codac

# Run your Python script
python ../train_offline.py --env hopper-medium-replay-v0 --wandb --num_quantiles 64 --seed 42