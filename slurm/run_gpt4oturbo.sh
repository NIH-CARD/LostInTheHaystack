#!/bin/bash

#SBATCH --job-name=gpt4oturbo
#SBATCH --mem=25g
#SBATCH --time=24:00:00
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Set up Conda
source ~/.bashrc
conda deactivate || true
conda activate streaming-env

# Run your script
# python -m scripts.run --exp-config configs/experiments/cbb_gpt4oturbo.yaml
python -m scripts.run --exp-config configs/experiments/or1m_gpt4oturbo.yaml
# python -m scripts.run --exp-config configs/experiments/nqkilt_gpt4oturbo.yaml