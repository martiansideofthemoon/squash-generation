#!/bin/sh
#SBATCH --job-name=job_squash_gpt2_corefs
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/logs/log_squash_gpt2_corefs_new_dir.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation

python -m torch.distributed.launch --nproc_per_node=4 question-generation/train.py --eval_before_start --n_epochs 4 --model_checkpoint gpt2 --dataset_path data/temp_dataset/instances_corefs --dataset_cache data/temp_dataset/cache_corefs --output_dir question-generation/gpt2_corefs_question_generation


