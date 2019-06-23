#!/bin/sh
#SBATCH --job-name=job_squash_gpt2
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/logs/log_squash_gpt2_new_dir.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation

python -m torch.distributed.launch --nproc_per_node=4 question-generation/train.py --eval_before_start --n_epochs 4 --model_checkpoint gpt2 --output_dir question-generation/gpt2_question_generation

