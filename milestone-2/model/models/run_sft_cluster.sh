#!/bin/bash -l
#SBATCH --chdir /scratch/izar/devries
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

# `SBATCH --something` is how you tell SLURM what resources you need
# The `--reservation cs-552` line only works during the 2-week period
# where 80 GPUs are available. Remove otherwise

python3 train_sft.py --max-length=512 --batch-size=2 --n-epochs=2 --logging-steps=10 --save-steps=500
