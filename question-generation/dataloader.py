import os
import logging
import pickle
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__file__)

class_map = {
    'causal': 'general',
    'judgemental': 'general',
    'instrumental': 'general',
    'general': 'general',
    'general_concept_completion': 'general',
    'specific': 'specific',
    'specific_concept_completion': 'specific'
}


def get_dataset(tokenizer, dataset_cache, path='data/temp_dataset/instances', split='train'):
    # Load question data
    dataset_cache = dataset_cache + '_' + split + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        return data

    dataset_path = "%s_%s.pickle" % (path, split)
    data = get_positional_dataset_from_file(tokenizer, file=dataset_path)

    if dataset_cache:
        torch.save(data, dataset_cache)

    logger.info("Dataset cached at %s", dataset_cache)

    return data


def get_dataset_from_file(tokenizer, file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    truncated_sequences = 0
    for inst in tqdm(data):
        tokenized_para = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        total_seq_len = len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4

        if total_seq_len > tokenizer.max_len:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_para = tokenized_para[:-1 * (total_seq_len - tokenizer.max_len + 1)]
            truncated_sequences += 1
            assert len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4 < tokenizer.max_len

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_para)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        inst['class'] = class_map[inst['class']]

    logger.info("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data


def get_position(para_ids, ans_ids, ans_prefix_ids):
    diff_index = -1
    # Find the first token where the paragraph and answer prefix differ
    for i, (pid, apid) in enumerate(zip(para_ids, ans_prefix_ids)):
        if pid != apid:
            diff_index = i
            break
    if diff_index == -1:
        diff_index = min(len(ans_prefix_ids), len(para_ids))
    # Starting from this token, we take a conservative overlap
    return (diff_index, min(diff_index + len(ans_ids), len(para_ids)))


def get_positional_dataset_from_file(tokenizer, file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    truncated_sequences = 0
    for inst in tqdm(data):
        tokenized_para = tokenizer.tokenize(inst['paragraph'])
        tokenized_question = tokenizer.tokenize(inst['question'])
        tokenized_answer = tokenizer.tokenize(inst['answer'])
        tokenized_ans_prefix = tokenizer.tokenize(inst['paragraph'][:inst['answer_position']])

        total_seq_len = len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4

        if total_seq_len > tokenizer.max_len:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_para = tokenized_para[:-1 * (total_seq_len - tokenizer.max_len + 1)]
            truncated_sequences += 1
            assert len(tokenized_para) + len(tokenized_answer) + len(tokenized_question) + 4 < tokenizer.max_len

        inst['paragraph'] = tokenizer.convert_tokens_to_ids(tokenized_para)
        inst['question'] = tokenizer.convert_tokens_to_ids(tokenized_question)
        inst['answer'] = tokenizer.convert_tokens_to_ids(tokenized_answer)
        ans_prefix_ids = tokenizer.convert_tokens_to_ids(tokenized_ans_prefix)
        inst['class'] = class_map[inst['class']]
        inst['answer_position_tokenized'] = get_position(inst['paragraph'], inst['answer'], ans_prefix_ids)
        pass

    logger.info("%d / %d sequences truncated due to positional embedding restriction" % (truncated_sequences, len(data)))

    return data
