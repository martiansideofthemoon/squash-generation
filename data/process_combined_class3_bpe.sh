#!/bin/bash
# Run this script from the data folder

echo "Normalizing and combining datasets ..."
python normalize_combined_class3.py
echo 'Learning BPE vocabulary ...'
python bert_bpe.py

mv train_questions_class.txt combined_class3/train_questions_class.txt
mv dev_questions_class.txt combined_class3/dev_questions_class.txt
mv train_questions_key.txt combined_class3/
mv dev_questions_key.txt combined_class3/

echo "Removing residue files ..."
rm corpus.txt dev_* train_*
