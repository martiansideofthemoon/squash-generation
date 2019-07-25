# Generating Question-Answer Hierarchies

This is the official repository accompanying the ACL 2019 long paper *[Generating Question-Answer Hierarchies](https://arxiv.org/abs/1906.02622)*. This repository contains the accompanying dataset and codebase. The code for the demo website can be found in [martiansideofthemoon/squash-website](https://github.com/martiansideofthemoon/squash-website). The demo is hosted [here](http://squash.cs.umass.edu:3000/?id=04bf8a42f934944809e76ec1).

The codebase in this repository contains a modified and improved version of the original codebase and tries to leverage language model pretraining in all its modules. The question generation module is a transformer-based model based off of GPT-2 which has been forked from [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai). The question answering module is a BERT-based SQUAD 2.0 model forked from [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT). Finally, the filtering has been greatly simplified and can be easily customized based on user preferences.

You can read a technical note on the modified system [here](https://arxiv.org/pdf/1906.02622.pdf#page=15).

## Requirements

Create a new Python 3.6 virtual environment. Run the following,

1. Install the requirements using `pip install -r requirements.txt`.

2. Install the English `spacy` library using `python -m spacy download en_core_web_sm`.

2. Since the code uses a slightly modified version of [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT), it needs to be installed locally. Run `cd pytorch-pretrained-BERT` followed by `pip install --editable .`

## Dataset

The training dataset for the question generation module can be found [here](https://drive.google.com/drive/folders/1FlVtPgyBiJIEOIecnNLH3cg0EbKkK0Z4?usp=sharing). This dataset contains QA from three reading comprehension datasets (SQuAD, CoQA and QuAC) labelled according to their conceptual category (as described in Table 1 of the paper). In addition, we have also provided the scheme that was adopted to label each question (hand labelling, rule-based templates or classifier. The distribution has been provided in Table A1 of the paper). These labels are finer than the classes used to train the models and contain an extra class (`verification`) for yes/no questions. The mapping to the coarse `general` and `specific` categories has been provided in [`question-generation/dataloader.py`](question-generation/dataloader.py#L11-L19).

#### Schema

A detailed schema for the original dataset has been provided in [`data/specificity_qa_dataset/README.md`](data/specificity_qa_dataset/README.md).

#### Preprocessing Instructions

During preprocessing, we remove generic, unanswerable, multi-paragraph and `verification` questions. Since coreferences in questions are common for the QuAC and CoQA datasets, we have an additional preprocessed version which resolves all the coreferences in the question statements.

0. Preprocessed versions of the dataset can be found in the same Google Drive [link](https://drive.google.com/drive/folders/1FlVtPgyBiJIEOIecnNLH3cg0EbKkK0Z4?usp=sharing). `instances_train.pickle` and `instances_dev.pickle` contain the standard filtered datasets. `instances_corefs_train.pickle` and `instances_corefs_dev.pickle` contain filtered datasets with question coreferences resolved. Place these files inside `data/temp_dataset`.

1. Download `train.pickle` and `dev.pickle` from the Google Drive [link](https://drive.google.com/drive/folders/1FlVtPgyBiJIEOIecnNLH3cg0EbKkK0Z4?usp=sharing) and place it in `data/specificity_qa_dataset`.

2. Run a filtering cycle using `python data/filter_dataset.py` to carry out standard filtering. Alternatively, you could run `python data/filter_dataset_corefs.py` to resolve coreferences in the questions in addition to filtering. Resolving coreferences can be resource and time intensive so you could use the preprocessed versions in the Google Drive link instead as described above.

#### Labelling Custom QA Datasets

We used the rules outlined in `data/question_rules.py` to carry out the rule-based labelling of questions. The classifier code was a simple 1-layer CNN built on top of ELMo embeddings (built using [allenai/allennlp](https://github.com/allenai/allennlp)) trained on hand-labelled questions. The code-base to classify questions has been added to `data/question-classifier`. The corresponding train/dev splits can be found under the `classifier` folder in the same Google Drive [link](https://drive.google.com/drive/folders/1FlVtPgyBiJIEOIecnNLH3cg0EbKkK0Z4?usp=sharing). Place the train and dev data in `data/question-classifier`. After training the model, run `python main.py` to train the classifier. You could use the `python main.py --mode classify` to classify new QA datasets.

Note that the classifier codebase is not production ready, so it might be a bit confusing and require modifications. (For instance, `general` questions are named `overview` and `specific` questions as `conceptual`).

You could alternatively try out BERT encoders using this [notebook](https://colab.research.google.com/drive/1Qvw5AJbcZXcPrkwM2nQp5zV00eM1aj0j) after modifying the `DatasetReader`.

## Question Generation

Our conditional question generation model is forked from [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai). We generate conditional questions using a language model which is fine-tuned from OpenAI's GPT or GPT2. We convert our training data as follows,

1. For `general` questions - `<bos> ... paragraph text ... <answer-general> ... answer span ... <question-general> ... question span ... <eos>`
2. For `specific` questions - `<bos> ... paragraph text ... <answer-specific> ... answer span ... <question-specific> ... question span ... <eos>`.

In addition, segmental embeddings are passed to the model (with specificity information) to provide a stronger signal about specificity of the question. A single language modelling objective is used to train the model optimized to minimize the loss on the question.

The codebase for the question generation module can be found under `question-generation`. Individual file descriptions have been added to [`question-generation/README.md`](question-generation/README.md).

[Slurm](https://slurm.schedmd.com/documentation.html) scheduler scripts have been provided under the `schedulers` folder in four different configurations. These scripts are bash scripts which also run without slurm. You will need to modify the `cd` command in these scripts to ensure you are in the current folder.

1. `schedulers/schedule_gpt.sh` - Fine-tune a question generation model starting from a pretrained GPT model.
2. `schedulers/schedule_gpt2.sh` - Fine-tune a question generation model starting from a pretrained GPT-2 model.
3. `schedulers/schedule_gpt_corefs.sh` - Fine-tune a question generation model on coref resolved data starting from a pretrained GPT model.
4. `schedulers/schedule_gpt2_corefs.sh` - Fine-tune a question generation model on coref resolved data starting from a pretrained GPT-2 model.

Since training the question generation model tends to be resource and time intensive, a pre-trained question generation model with the `schedulers/schedule_gpt2_corefs.sh` configuration has been released [here](https://drive.google.com/drive/folders/1HEbm_sHDAAcylKIF4vIvZ9N2jEA7I5Em?usp=sharing).

Extract the pre-trained question generation model in the folder `question-generation/gpt2_corefs_question_generation`.

## Question Answering

Our question answering module is a BERT-based model trained on SQuAD 2.0, forked from [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT). The codebase for the question answering module can be found under `question-answering`. Individual file descriptions have been added to [`question-answering/README.md`](question-answering/README.md).

[Slurm](https://slurm.schedmd.com/documentation.html) scheduler scripts have been provided under the `schedulers` folder in two different configurations. These scripts are bash scripts which also run without slurm. You will need to modify the `cd` command in these scripts to ensure you are in the current folder.

1. `schedulers/schedule_squad_bert.sh` - Run a BERT-base model on SQuAD 2.0
2. `schedulers/schedule_squad_bert_large.sh` - Run a BERT-large model on SQuAD 2.0

Since training the QA model tends to be resource and time intensive, a pre-trained QA model using the `schedulers/schedule_squad_bert_large.sh` configuration has been released [here](https://drive.google.com/drive/folders/1D3fIPuwn0C0zIMg29QSKcnSAc8HfNemd?usp=sharing). This model gets an F1 score of 78.8 on the SQuAD 2.0 development set (the original BERT paper reports an F1 score of 81.9).

Extract the pre-trained QA model in the folder `question-answering/bert_large_qa_model`.

## SQUASHing

Once the question generation and question answering modules have been trained, run `squash/pipeline.sh` to choose an arbitary development set example from QuAC and SQUASH it. You might need to modify the model checkpoint directories for the question generation or question answering module. The output document will be available in `squash/final/`. Individual file descriptions have been added to [`squash/README.md`](squash/README.md).

For custom inputs, make a folder `squash/temp/$KEY` where `$KEY` is a unique identifier. Additionally, you will need to create a `squash/temp/$KEY/metadata.json` file to specify the settings and input text. For an example, look at [`squash/temp/quac_869/metadata.json`](squash/temp/quac_869/metadata.json). Finally run `squash/pipeline_custom.sh $KEY`.

## Citation

If you find this code or dataset useful, please cite us.

```
@inproceedings{squash2019,
Author = {Kalpesh Krishna and Mohit Iyyer},
Booktitle = {Association for Computational Linguistics,
Year = "2019",
Title = {Generating Question-Answer Hierarchies}
}
```
