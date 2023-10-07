#!/usr/bin/env bash
#SBATCH --job-name=CUDA_Example
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=dc007
#SBATCH --output=/work/dc007/dc007/cs-rami1/cache_llm/output/%j.out
#module load anaconda/python3 
#module load python/3.8.16-gpu
#module load pytorch
#export PYTHONUSERBASE=/work/dc007/dc007/cs-rami1/cache
#export PATH=${PYTHONUSERBASE}/bin:${PATH}
#export PYTHONPATH=${PYTHONUSERBdASE}/lib/${MINICONDA3_PYTHON_LABEL}/site-packages:${PYTHONPATH}
export TRANSFORMERS_CACHE=/work/dc007/dc007/cs-rami1/.cache/huggingface/hub
export NEPTUNE_MODE='offline'
#source /work/dc007/dc007/cs-rami1/cache/bin/
## sembla que funciona??? /mnt/lustre/indy2lfs/work/dc007/dc007/cs-rami1/cache/bin/python
##conda activate /work/dc007/dc007/cs-rami1/conda_envs/cache
echo $TAGS
cd /work/dc007/dc007/cs-rami1/cache_llm
for SEED in 0 1 2
do
    export SEED
    export BUDGET=600,800,1000,1500,2000
    bash scripts/run_2.sh
    export BUDGET=2500
    bash scripts/run_2.sh
    export BUDGET=3000
    bash scripts/run_2.sh
    export BUDGET=3500
    bash scripts/run_2.sh
done

#conda activate conda_envs/test
#export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZWEyZjY5NS05NGFjLTQ5YTItYTZlYS04ZjA1NGZkNGY2NWQifQ==
#neptune sync -p cache/cache-llm