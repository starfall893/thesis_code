#!/bin/bash
#SBATCH --job-name=thesis_opt
#SBATCH --output=thesis_opt_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lr3956@princeton.edu

# Load modules
module purge
module load anaconda3/2024.6

# Activate conda environment
conda activate base  # or your specific environment name

# Print GPU information
nvidia-smi

# Run the main script
python main.py

# Print completion message
echo "Job completed at $(date)" 