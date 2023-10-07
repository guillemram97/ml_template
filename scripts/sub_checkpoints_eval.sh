#!/usr/bin/env bash
#SBATCH --qos epsrc
#SBATCH --time 24:0:0 --tasks-per-node 1
#SBATCH -N 1 -n 1 --mem-per-gpu 122G
#SBATCH --gres gpu:1 --cpus-per-task 144
#SBATCH --output=/bask/projects/x/xngs6460-languages/guillem/cache_llm/output/%j.out
#SBATCH --constraint=a100_40
module purge
module load baskerville
source /bask/apps/live/EL8-ice/software/Miniconda3/4.10.3/etc/profile.d/conda.sh
conda init bash
conda activate cache

cd /bask/projects/x/xngs6460-languages/guillem/cache_llm

for TASK in ag_news mmlu-ss
do
    for CHECKPOINT in 600 700 800 900 1000 1600 1700 1800 1900 2000 2600 2700 2800 2900 3000
    do
        python checkpoints_eval.py --task_name $TASK --checkpoint $CHECKPOINT
    done
done