#!/bin/bash
#
#SBATCH --job-name=job_doc2qa
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/logs/log_doc2qa.txt
#SBATCH --time=24:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --open-mode append
#SBATCH -d singleton

while :
do
	echo "Waiting in the queue..."

	KEY=$(python squash/wait_queue.py)

	echo $KEY

	echo 'Extracting answers ...'
	python squash/extract_answers.py --key $KEY

	echo 'Generating questions ...'
	python question-generation/interact.py \
		--model_checkpoint question-generation/gpt2_corefs_question_generation \
		--model_type gpt2 \
		--key $KEY

	echo 'Running QA module ...'
	python question-answering/run_squad.py \
		--bert_model question-answering//bert_large_qa_model \
		--do_predict \
		--do_lower_case \
		--predict_file squash/temp/$KEY/generated_questions.json \
		--output_dir squash/temp/$KEY \
		--predict_batch_size 16 \
	  	--version_2_with_negative

	 echo 'Combining Q and A ...'
	 python squash/combine_qa.py --key $KEY

	echo 'Filtering bad Q/As ...'
	python squash/filter.py --key $KEY

	echo 'Popping top value from queue ...'
	sed -i '1d' squash/temp/queue.txt

done