#!/bin/bash
#SBATCH --job-name=ant_obstacle_offline     # Job name
#SBATCH --partition=gpunodes          # Partition to use (adjust if needed)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory allocation
#SBATCH --time=08:00:00               # Maximum runtime (8 hour)
#SBATCH --output=ant_obstacle_offline.out  # Standard output file
#SBATCH --error=ant_obstacle_offline.err   # Standard error file

# Activate your Conda environment
# conda activate codac

# Run your Python script
# python ../train_online.py --env AntObstacle-v0 --risk_prob 0.95 --risk_penalty 90 --algo codac --risk_type neutral --entropy true --wandb
python ../train_offline.py --env AntObstacle-v0 --risk_prob 0.95 --risk_penalty 90 --algo codac --risk_type cvar --entropy true --dist_penalty_type uniform --min_z_weight 0.1 --lag 10.0 --dataset_epoch 5000 --seed 0 --wandb
