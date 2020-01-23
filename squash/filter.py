import argparse
import numpy as np
import json
import pickle
import random
import sys
import os
import spacy
import time

from collections import defaultdict

from squad_eval_utils import (
    f1_metric,
    exact_match_metric,
    metric_max_over_candidates,
    recall_metric,
    precision_metric,
    normalize
)

parser = argparse.ArgumentParser()

parser.add_argument('--key', type=str, default=None,
                    help='Key used to uniquely identify the files needed.')

args = parser.parse_args()

nlp = spacy.load('en_core_web_sm')


class Paragraph(object):
    def __init__(self, para, filter_frac={'general_sent': 0.5, 'specific_sent': 0.8, 'specific_entity': 0.8}):
        self.text = para['context']
        self.original_qas = para['qas']
        self.sentences = []
        # For every QA candidate, calculate the overlap scores between answering module and original question
        self.calculate_overlap_scores()
        # Count total pool of general and specific questions initially present
        total = {
            'general_sent': self.count(self.original_qas, 'general_sent'),
            'specific_sent': self.count(self.original_qas, 'specific_sent'),
            'specific_entity': self.count(self.original_qas, 'specific_entity'),
        }
        # Calculate the expected number of questions of each type
        expected = {
            k: max(int(np.ceil(filter_frac[k] * total[k])), 1) for k, _ in total.items()
        }

        len_original_qas = len(self.original_qas)
        # Remove exact duplicate question text candidates
        unique_qs = self.remove_exact_duplicate_questions(self.original_qas)
        len_unique_qs = len(unique_qs)
        # Within every paragraph, remove exact-duplicate predicted answer texts
        unique_qas = self.remove_exact_duplicate_answers(unique_qs, "specific")
        unique_qas = self.remove_exact_duplicate_answers(unique_qas, "general")

        print("%d / %d left after removing exact duplicate questions" % (len_unique_qs, len_original_qas))
        print("%d / %d left after removing exact duplicate answers" % (len(unique_qas), len_original_qas))

        # From the remaining candidates, carry out filtering based on filter_frac
        # For filtering, remove unanswerable questions, generic and bad entity questions and finally
        # sort by answer overlap questions
        filtered_qas = self.filter(unique_qas, expected)
        self.binned_qas = self.bin(filtered_qas)

    def count(self, input_list, question_algo):
        return sum([1 for qa in input_list if qa['algorithm'] == question_algo])

    def calculate_overlap_scores(self):
        for qa in self.original_qas:
            ans = qa['predicted_answer']
            gt_ans = qa['answers'][0]['text']
            # store all word overlap metrics between predicted answer and original answer
            qa['exact_match'] = exact_match_metric(ans, gt_ans)
            qa['f1_match'] = f1_metric(ans, gt_ans)
            qa['recall_match'] = recall_metric(ans, gt_ans)
            qa['precision_match'] = precision_metric(ans, gt_ans)
            qa['unanswerable'] = qa['predicted_answer'] == ''

    def remove_exact_duplicate_questions(self, input_list):
        unique_qa_list = []
        unique_qa_dict = defaultdict(list)

        for qa in input_list:
            normalized_q = normalize(qa['question'])
            unique_qa_dict[normalized_q].append(qa)

        # Keep only first occurence for each question
        # Since answering model is seeing same (paragraph, question) input, answer will be same
        for _, duplicates in unique_qa_dict.items():
            unique_qa_list.append(duplicates[0])

        unique_qa_list.sort(key=lambda x: x['id'])

        return unique_qa_list

    def remove_exact_duplicate_answers(self, input_list, question_type="specific"):
        """Within each paragraph, ensure that every GENERAL / SPECIFIC predicted answer is unique."""
        unique_qa_list = []
        unique_qa_dict = defaultdict(list)

        if question_type == "specific":
            retain_type = "general"
        else:
            retain_type = "specific"

        for qa in input_list:
            # We don't want to be too aggressive about the unanswerable questions since the
            # answering module could have made mistakes. Unanswerable questions will be dealt
            # with later on in the filtering phase depending on the user settings.
            if qa['unanswerable'] is True or retain_type in qa["algorithm"]:
                unique_qa_list.append(qa)
                continue
            normalized_a = normalize(qa['predicted_answer'])
            unique_qa_dict[normalized_a].append(qa)

        metrics = {
            "general_sent": "recall_match",
            "specific_sent": "precision_match",
            "specific_entity": "recall_match"
        }

        # Now, sort the duplications according to the precision / recall scores with intended answer
        for _, duplicates in unique_qa_dict.items():
            duplicates.sort(key=lambda x: (-x[metrics[x['algorithm']]], x['id']))
            unique_qa_list.append(duplicates[0])

        unique_qa_list.sort(key=lambda x: x['id'])

        return unique_qa_list

    def filter(self, qa_list, expected):

        filtered_qa = []

        def filter_fn(algorithm, metric):
            """Generic filtering function for each question type"""
            relevant_qa = [qa for qa in qa_list if qa['algorithm'] == algorithm]
            relevant_qa.sort(key=lambda x: (-x[metric], x['id']))
            relevant_qa = relevant_qa[:expected[algorithm]]
            relevant_qa.sort(key=lambda x: x['id'])
            return relevant_qa

        # Filter out bad GENERAL questions
        filtered_qa = \
            filter_fn('general_sent', 'recall_match') + \
            filter_fn('specific_sent', 'precision_match') + \
            filter_fn('specific_entity', 'recall_match')

        return filtered_qa

    def bin(self, filtered_qas):
        general_questions = [qa for qa in filtered_qas if qa['algorithm'] == 'general_sent']
        specific_questions = [qa for qa in filtered_qas if 'specific' in qa['algorithm']]

        for qa in general_questions:
            qa['children'] = []

        # find general parent for every specific question
        def find_parent(sq):
            # find precision of predicted answer with reference general question answer
            gq_precisions = [
                precision_metric(sq['predicted_answer'], gq['answers'][0]['text'])
                for gq in general_questions
            ]
            if np.max(gq_precisions) > 0.5:
                # precision is high enough, return the top point
                return np.argmax(gq_precisions)
            else:
                # if precision is too low, resort to the answer positioning heuristic
                current_pos = sq['answers'][0]['answer_start']
                for i, gq in enumerate(general_questions):
                    if gq['answers'][0]['answer_start'] > current_pos:
                        return i - 1
                return len(general_questions) - 1

        for sq in specific_questions:
            gq_parent = find_parent(sq)
            general_questions[gq_parent]['children'].append(sq)

        # sort the specific questions by their answer position in the paragraph
        # this is necessary since specific QA need not be in sorted order
        for qa in general_questions:
            qa['children'].sort(key=lambda x: x['answers'][0]['answer_start'])

        return general_questions


with open('squash/temp/%s/final_qa_set.json' % args.key, 'r') as f:
    paras = json.loads(f.read())['data'][0]['paragraphs']

with open("squash/temp/%s/metadata.json" % args.key, "r") as f:
    metadata = json.loads(f.read())

filter_frac = {
    "general_sent": metadata["settings"]["gen_frac"],
    "specific_sent": metadata["settings"]["spec_frac"],
    "specific_entity": metadata["settings"]["spec_frac"]
}

full_summary = [Paragraph(para, filter_frac=filter_frac) for para in paras]

squash_output = {
    'instance_key': args.key,
    'qa_tree': []
}

for para in full_summary:
    squash_output['qa_tree'].append({
        'para_text': para.text,
        'binned_qas': para.binned_qas
    })

with open('squash/final/%s.json' % args.key, 'w') as f:
    f.write(json.dumps(squash_output))
