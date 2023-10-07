#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=14000 
#SBATCH --cpus-per-task=4
#SBATCH --partition=CDT_GPU
source /home/${USER}/miniconda3/bin/activate cache
rsync data /home/ to /disk/scratch/
bash scripts/run.sh
rm -rf /disk/scratch/${USER}/exp