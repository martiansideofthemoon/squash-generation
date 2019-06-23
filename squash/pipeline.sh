#!/bin/bash
#
#SBATCH --job-name=job_doc2qa
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/qa-generation/logs/log_doc2qa.txt
#SBATCH --time=24:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --open-mode append
#SBATCH -d singleton

dataset=combined_class3
export SQUAD_DIR=/mnt/nfs/work1/miyyer/datasets/SQuAD


for var in {1..10}
do
	echo "Run $var ..."
	# assume input in doc2qa/input.txt in the form of a set of paragraphs
	# alternatively, we can populate the file with a random input from the dev set
	echo 'Choosing instance from QuAC dev set ...'
	python squash/populate_input.py

	echo 'Extracting answers ...'
	python squash/extract_answers.py

	echo 'Generating questions ...'
	python src/interact.py \
		--model_checkpoint runs/May26_09-29-23_node009 \
		--filename squash/temp/input.pkl \
		--model_type gpt2

	echo 'Running QA module ...'
	python qa_model/run_squad.py \
		--bert_model qa_model/save_large \
		--do_predict \
		--do_lower_case \
		--predict_file squash/temp/generated_questions.json \
		--output_dir squash/temp \
	  	--version_2_with_negative

	 echo 'Combining Q and A ...'
	 python squash/combine_qa.py

	echo 'Filtering bad Q/As ...'
	python squash/filter.py

done