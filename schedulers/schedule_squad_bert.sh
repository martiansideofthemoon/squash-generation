#!/bin/sh
#SBATCH --job-name=job_squad_bert
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/logs/log_squad_bert.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=45GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation

export SQUAD_DIR=/mnt/nfs/work1/miyyer/datasets/SQuAD

python question-answering/run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir question-answering/bert_base_qa_model \
  --version_2_with_negative
