#!/bin/bash
#SBATCH -J all-bn		# job name
#SBATCH -o all-bn_%j		# stdout
#SBATCH -p gh                   # GH (Grace-Hopper) GPU partition
#SBATCH -N 1                    # 1 node
#SBATCH -t 16:00:00             # 8 hour wall-clock limit

echo "Entered Training Script..."

# Source
source ~/.bashrc

# Load modules
module load gcc/13.2.0 cuda/12.4 nvidia_math/12.4 cmake/3.29.5

# Activate environment
conda activate crecs

# Enter the project
cd ~/work/C-RECS/

# Start the program
python train.py --config configs/content.yaml
