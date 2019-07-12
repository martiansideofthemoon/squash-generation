## SQUASHing

This is the codebase for the SQUASH pipeline which uses pretrained question generation and question answering modules to converting input paragraphs into trees of question-answer pairs.

## File Descriptions

1. `populate_input.py` - Choose a random instance from QuAC for SQUASHing.
2. `extract_answers.py` - Extract individual sentences and entities which will be used for question generation.
3. `combine_qa.py` - Module to combine generated questions with outputs from the question answering module.
4. `filter.py` - Filter the pool of generated question-answer pairs and carry out binning.
5. `squad_eval_utils.py` - Utilities to provide F1 overlap between two answer spans, used for the filtering process.
6. `pipeline.sh` - Run the whole SQUASH pipeline.
7. `pipeline_custom.sh` - Run the SQUASH pipeline with custom inputs specified as the first command line argument. Make sure you have create `squash/temp/$KEY` and `squash/temp/$KEY/metadata.json` before running this.
