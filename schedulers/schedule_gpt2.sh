#!/bin/sh
#SBATCH --job-name=job_squash_gpt2
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/logs/log_squash_gpt2.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation

python -m torch.distributed.launch --nproc_per_node=4 question-generation/train.py --eval_before_start --n_epochs 5 --model_checkpoint gpt2

