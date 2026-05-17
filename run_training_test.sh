#!/bin/bash
#SBATCH --job-name=review_class
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_bscls
#SBATCH --output=logs/rev_%j.out  
#SBATCH --error=logs/rev_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=03:00:00  

# activate venv
source .venv/bin/activate

python -u training_pipeline.py --model_name "deberta-v3-base"
