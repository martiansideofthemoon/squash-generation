""" Official evaluation script for v1.1 of the SQuAD dataset. """
# pylint: skip-file
from __future__ import print_function
from collections import Counter
import numpy as np
import string
import re
import argparse
import json
import sys


def normalize(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_metric(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_metric(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    return precision


def recall_metric(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall


def exact_match_metric(prediction, ground_truth):
    return (normalize(prediction) == normalize(ground_truth))


def metric_max_over_candidates(metric_fn, candidates, ground_truth):
    scores_for_candidates = []
    for candidate in candidates:
        score = metric_fn(candidate, ground_truth)
        scores_for_candidates.append(score)
    return max(scores_for_candidates), np.argmax(scores_for_candidates), scores_for_candidates
