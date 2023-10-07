#!/usr/bin/env bash
#SBATCH --qos epsrc
#SBATCH --time 12:00:0 --tasks-per-node 1
#SBATCH -N 1 -n 1 --mem-per-gpu 122G
#SBATCH --gres gpu:1 --cpus-per-task 144
#SBATCH --output=/bask/projects/x/xngs6460-languages/guillem/cache_llm/output/%j.out
#######SBATCH --constraint=a100_40
module purge
module load baskerville
source /bask/apps/live/EL8-ice/software/Miniconda3/4.10.3/etc/profile.d/conda.sh
conda init bash
conda activate cache

cd /bask/projects/x/xngs6460-languages/guillem/cache_llm


export BUDGET
bash scripts/run.sh