# Generating Question-Answer Hierarchies

This is the official repository accompanying the ACL 2019 long paper *[Generating Question-Answer Hierarchies](https://arxiv.org/abs/1906.02622)*. This repository contains the accompanying dataset and codebase. The code for the demo website can be found [here](https://github.com/martiansideofthemoon/squash-website).

## Dataset

The training dataset for the question generation module can be found [here](https://drive.google.com/open?id=1FlVtPgyBiJIEOIecnNLH3cg0EbKkK0Z4). This dataset contains QA from three reading comprehension datasets (SQuAD, CoQA and QuAC) labelled according to their conceptual category (as described in Table 1 of the paper). In addition, we have also provided the scheme that was adopted to label each question (hand labelling, rule-based templates or classifier. The distribution has been provided in Table A1 of the paper). These labels are finer than the classes used to train the models and contain an extra class (`verification`) for yes/no questions. The mapping to the coarse `general` and `specific` categories has been provided in [`src/dataloader.py`](https://github.com/martiansideofthemoon/squash-generation/blob/master/src/dataloader.py#L11-L19).

#### Schema

A detailed schema for the original dataset has been provided in [`data/specificity_qa_dataset/README.md`](https://github.com/martiansideofthemoon/squash-generation/blob/master/data/specificity_qa_dataset/README.md).

#### Preprocessing Instructions

During preprocessing, we remove generic, unanswerable, multi-paragraph and `verification` questions. Since coreferences in questions are common for the QuAC and CoQA datasets, we have an additional preprocessed version which resolves all the coreferences in the question statements.

0. Preprocessed versions of the dataset can be found in the Google Drive link. `instances_train.pickle` and `instances_dev.pickle` contain the standard filtered datasets. `instances_corefs_train.pickle` and `instances_corefs_dev.pickle` contain filtered datasets with question coreferences resolved. Place these files inside `data/temp_dataset`.

1. Download `train.pickle` and `dev.pickle` from the Google Drive link and place it in `data/specificity_qa_dataset`.

2. Run a filtering cycle using `python data/filter_dataset.py` to carry out standard filtering. Alternatively, you could run `python data/filter_dataset_corefs.py` to resolve coreferences in the questions in addition to filtering. Resolving coreferences can be resource and time intensive so you could use the preprocessed versions in the Google Drive link instead as described above.