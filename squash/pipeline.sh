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

for var in {1..1}
do
	echo "Run $var ..."
	echo 'Choosing instance from QuAC dev set ...'
	python squash/populate_input.py

	echo 'Extracting answers ...'
	python squash/extract_answers.py

	echo 'Generating questions ...'
	python question-generation/interact.py \
		--model_checkpoint question-generation/gpt2_corefs_question_generation \
		--filename squash/temp/input.pkl \
		--model_type gpt2

	echo 'Running QA module ...'
	python question-answering/run_squad.py \
		--bert_model question-answering//bert_large_qa_model \
		--do_predict \
		--do_lower_case \
		--predict_file squash/temp/generated_questions.json \
		--output_dir squash/temp \
		--predict_batch_size 16 \
	  	--version_2_with_negative

	 echo 'Combining Q and A ...'
	 python squash/combine_qa.py

	echo 'Filtering bad Q/As ...'
	python squash/filter.py

done