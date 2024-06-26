#!/bin/bash
#SBATCH --job-name=run_job      # Job name
#SBATCH --partition=gpu             # Partition name
#SBATCH --gres=gpu:1                # Request one GPU
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --time=24:00:00             # Time limit (hh:mm:ss)
#SBATCH --output=slurm/logs/output_%j.log      # Standard output and error log (with job ID)
#SBATCH --mem=200G                   # Memory per node

# Load the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# python /home/dalnuhait/Aurora-GPT-PL/pipeline.py
python /home/dalnuhait/Aurora-GPT-PL/src/dpo.py --model_id "meta-llama/Llama-2-7b-hf" --dataset_name ultrafeedback