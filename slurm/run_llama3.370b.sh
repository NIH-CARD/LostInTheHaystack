#!/bin/bash

#SBATCH --job-name=llama3.370b
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
python -m scripts.run --exp-config configs/experiments/cbb_llama3.370b.yaml
python -m scripts.run --exp-config configs/experiments/nm_llama3.370b.yaml
python -m scripts.run --exp-config configs/experiments/nq_llama3.370b.yaml
