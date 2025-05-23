#!/bin/bash

#SBATCH --job-name=llama8B
#SBATCH --mem=100g
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Set up Conda
source ~/.bashrc
conda deactivate || true
conda activate streaming-env

# Run your script
# python -m scripts.run --exp-config configs/experiments/cbb_llama8b.yaml
python -m scripts.run --exp-config configs/experiments/or1m_llama8b.yaml
# python -m scripts.run --exp-config configs/experiments/nqkilt_llama8b.yaml