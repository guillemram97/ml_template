#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/rds/user/cs-rami1/rds-t2-cs119/guillem/cache_llm/output/%j.out

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source ~/.bashrc
conda activate cache

cd /home/$USER/cache_llm


#export TRAIN_SAMPLES=10000
#export RETRAIN_FREQ=100
export TARGET=llm
export BASE_MODEL=t5-base
export PART=csd3
export DATA_PATH=/rds/user/cs-rami1/rds-t2-cs119/guillem/datasets/data
#export SOFT_LABELS=1

#i=-1
########
################################
#export TASK_NAME=isear
################################
#export CHECKPOINT=2000
#export BUDGET=1000
#export SEED=0
#export TAGS=$BASE_MODEL,$TASK_NAME,$STRATEGY,$TARGET,$PART,NO_RETRAIN
#i=$((i+1))
#bash scripts/run_no_retrain.sh &
#pids[${i}]=$!

for BUDGET in 1000 1500 2000 2500 3000 3500
do
    export BUDGET
    bash scripts/run.sh
done


#for pid in ${pids[*]}; do
#    wait $pid
#done

