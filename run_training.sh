#!/bin/bash
#SBATCH --job-name=review_class
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs/rev_%A_%a.out  # %A is master job ID, %a is array task ID
#SBATCH --error=logs/rev_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=2:00:00  
#SBATCH --array=0-3  # spawn 4 jobs at once

# activate venv
source .venv/bin/activate

models=("Bio_ClinicalBERT" "deberta-v3-base" "roberta-base" "sentiment-roberta-large-english")

# map idx for specific job
model_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))

# extract specific variable for job
model=${models[$model_idx]}

echo "Running task ID $SLURM_ARRAY_TASK_ID -> Model: $model "

# run pipeline with set arguments
python -u training_pipeline.py --model_name "$model"
