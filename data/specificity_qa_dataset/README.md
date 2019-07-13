## Dataset for Generating Question-Answer Hierarchies

The training dataset for the question generation module can be found [here](https://drive.google.com/open?id=1FlVtPgyBiJIEOIecnNLH3cg0EbKkK0Z4). This dataset contains QA from three reading comprehension datasets (SQuAD, CoQA and QuAC) labelled according to their conceptual category (as described in Table 1 of the paper). In addition, we have also provided the scheme that was adopted to label each question (hand labelling, rule-based templates or classifier. The distribution has been provided in Table A1 of the paper). These labels are finer than the classes used to train the models and contain an extra class (`verification`) for yes/no questions. The mapping to the coarse `general` and `specific` categories has been provided in [`src/dataloader.py`](https://github.com/martiansideofthemoon/squash-generation/blob/master/src/dataloader.py#L11-L19).

This directory is supposed to contain the raw `train.pickle` and `dev.pickle` files. The schema for the data has been described below.

#### Schema

The dataset is a list of instances. In the first level,

1. `article_title` - Article title take from original dataset.
2. `section_title` - Section title take from original dataset.
3. `context` - Multi-paragraph section which acts like the context for reading comprehension dataset.
4. `num_paras` - Number of paragraphs in the dataset.
5. `dataset` - The dataset from which this instance was taken.
6. `sequence` - The instance number in the original dataset.
7. `unanswered` - Information about the unanswerable questions in this instance. Ignored during preprocessing.

Finally, `paragraphs` breaks down each multi-paragraph instance and splits up the questions by paragraph according to their answer position.

8. `paragraphs/<item>/text` - Text for the individual paragraph.
9. `paragraphs/<item>/qas/<item>/question` - Text for the question about the paragraph.
10. `paragraphs/<item>/qas/<item>/answer` - Answer span for the question from the paragraph.
11. `paragraphs/<item>/qas/<item>/qa_num` - Specifies the question number in the original dataset.
12. `paragraphs/<item>/qas/<item>/global_ans_position` - Where does the answer span start in the original QuAC / CoQA context? (absent in SQUAD instances)
13. `paragraphs/<item>/qas/<item>/local_ans_position` - Where does the answer span start in the current paragraph?
13. `paragraphs/<item>/qas/<item>/partial` - Specifices whether the answer span is fully contained in the paragraph. If `(FULL)`, the answer is completely inside the paragraph. If `(PARTIAL)`, the answer spans multiple paragraphs. If `(UNANSWERED)`, the answer length was zero.
14. `paragraphs/<item>/qas/<item>/conceptual_category` - The Lehnert 1978 conceptual category assigned by our labelling scheme. Can be `instrumental`, `causal`, `general_concept_completion`, `specific_concept_completion`, `verification`, `judgmental`. The mapping to `general` and `specific` has been discussed earlier in the README.
15. `paragraphs/<item>/qas/<item>/labelling_scheme` - Labelling scheme used to annotate the conceptual category of this question. Can be `hand`, `rules` or `classifier`.
16. `paragraphs/<item>/qas/<item>/followup` (only for QuAC) - Followup information for the QuAC dataset.
