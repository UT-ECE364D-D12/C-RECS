#!/bin/bash
#SBATCH -J descriptions		# job name
#SBATCH -o descriptions_%j	# stdout
#SBATCH -p gh                   # GH (Grace-Hopper) GPU partition
#SBATCH -N 1                    # 1 node
#SBATCH -t 08:00:00             # 8 hour wall-clock limit

# Source
source ~/.bashrc

# Load modules
module load gcc/13.2.0 cuda/12.4 nvidia_math/12.4 cmake/3.29.5

# Activate environment
conda activate crecs

# Enter the project
cd ~/work/C-RECS/

# Start the program
python scripts/simulate_ml-20m.py ~/work/datasets/crecs/ml-20m/ --mode=descriptions --max_length=256
