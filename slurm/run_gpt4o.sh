#!/bin/bash

#SBATCH --job-name=gpt4o
#SBATCH --mem=25g
#SBATCH --time=48:00:00
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Set up Conda
source ~/.bashrc
conda deactivate || true
conda activate streaming-env

# Run your script
python -m scripts.run --exp-config configs/experiments/cbb_gpt4o.yaml
python -m scripts.run --exp-config configs/experiments/nm_gpt4o.yaml
python -m scripts.run --exp-config configs/experiments/nq_gpt4o.yaml