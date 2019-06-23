## Dataset Preprocessing

## File Descriptions

1. `question_rules.py` - Rules used for labelling questions according to their specificity.
2. `filter_dataset.py` - Script to preprocess dataset, filtering out generic, unanswerable, multi-paragraph and `verification` questions.
3. `filter_dataset_corefs.py` - Script to preprocess dataset, filtering out generic, unanswerable, multi-paragraph and `verification` questions. In addition, question corefs are resolved.
4. `blacklist.py` - A hand-curated list of generic questions commonly found in `QuAC`.
5. `question-classifier` - Classifier to label questions according to their specificity.
