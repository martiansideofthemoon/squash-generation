#!/bin/sh
#SBATCH --job-name=job_data_coref
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/logs/log_data_coref.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=370GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation

python data/filter_dataset_corefs.py
