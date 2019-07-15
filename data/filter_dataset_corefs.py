"""Fork from the data/filter_dataset.py file. In addition to filtering the raw dataset,
this script resolves unresolvable coreferences from the questions (very common in QuAC and CoQA)"""

import pickle
import unicodedata
import re
import sys

import spacy
import neuralcoref

from tqdm import tqdm
from collections import Counter

from multiprocessing import Pool
from blacklist import black_list, black_list_patterns


DOWNSAMPLE_THRESHOLD = 10
# Make sure your machine has atleast NUM_PROCESSES CPU cores
NUM_PROCESSES = 48
# Each chunk is processed batch-wise by Spacy.
# Higher chunk sizes lead to faster pre-processing but an unresponsive progressbar
CHUNK_SIZE = 5


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(string):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    string = re.sub(r"\'", " \'", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def coref_worker(para_questions):
    # https://github.com/explosion/spaCy/issues/1839#issuecomment-443227516
    # Avoids serialization of the nlp object
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp, blacklist=False)

    docs = list(nlp.pipe(para_questions))

    all_coref_resolved_questions = []

    for doc in docs:
        # Isolate the question only
        last_tilda = 0
        for token_index, token in enumerate(doc):
            if token.text == '~':
                last_tilda = token_index

        question_span = doc[last_tilda + 1:]

        new_question = []
        corefs_used = {}

        # This loop runs over each token in the question and replaces unresolved corefs.
        # Unresolved corefs are common in QuAC and CoQA.
        # Corefs were also resolved in paragraphs in https://arxiv.org/abs/1805.05942 for question generation
        for token in question_span:
            if token._.in_coref is False or \
               token.pos_ not in ['PRON', 'DET'] or \
               token.text == 'the':
                new_question.append(token.text_with_ws)
            else:
                cluster = token._.coref_clusters[0]
                if cluster.i in corefs_used or \
                   len(cluster.main.text.split()) > 5 or \
                   (len(cluster.main) == 1 and cluster.main[0].pos_ in ['PRON', 'DET']):
                    new_question.append(token.text_with_ws)
                else:
                    corefs_used[cluster.i] = 1
                    new_question.append(cluster.main.text + ' ')

        all_coref_resolved_questions.append(''.join(new_question))

    return all_coref_resolved_questions


for corpus in ['train', 'dev']:
    print("\nSplit = %s" % corpus)
    with open('data/specificity_qa_dataset/%s.pickle' % corpus, 'rb') as f:
        data = pickle.load(f)

    partial_answers = 0
    zero_length_answers = 0
    black_listed = 0
    inst_num = 0

    all_para_text = []
    all_para_questions = []
    all_conceptual_category = []
    all_answer = []
    all_ans_positions = []

    for instance in tqdm(data):

        for para_num, para in enumerate(instance['paragraphs']):

            for qa in para['qas']:

                # Remove answers spanning multiple paragraphs for now
                if qa['partial'] != '(FULL)':
                    partial_answers += 1
                    continue

                # Remove answers which won't have any tokens after normalization
                if normalize_string(qa['answer']) == "":
                    zero_length_answers += 1
                    continue

                # Remove generic questions which have exact matches in the blacklist
                if qa['question'].lower() in black_list:
                    black_listed += 1
                    continue

                # Remove verification questions as they cannot be reliably mapped to a specificity
                if qa['conceptual_category'] == 'verification':
                    continue

                # Remove generic questions with regex matches in the blacklist
                pattern_match_found = False
                for pattern in black_list_patterns:
                    if len(re.findall(pattern, qa['question'].lower())) > 0:
                        pattern_match_found = True

                if pattern_match_found is True:
                    black_listed += 1
                    continue

                assert para['text'][qa['local_ans_position']:qa['local_ans_position'] + len(qa['answer'])] == qa['answer']

                # Concatenate paragraph with question with a separator token for coreference resolution in question
                # Since SQuAD instances are independently written per-paragraph, only look at current paragraph
                if instance["dataset"] == "squad":
                    current_para_text = para["text"]
                else:
                    current_para_text = ' '.join([x['text'] for x in instance['paragraphs'][:para_num + 1]])

                para_question = current_para_text.strip() + " ~ " + qa['question'].strip()

                all_para_text.append(para['text'])
                all_para_questions.append(para_question)
                all_conceptual_category.append(qa['conceptual_category'])
                all_answer.append(qa['answer'])
                all_ans_positions.append(qa['local_ans_position'])

    pool = Pool(processes=NUM_PROCESSES)

    # chunk questions into blocks of chunk_size
    all_para_questions_chunked = []
    for i in range(0, len(all_para_questions), CHUNK_SIZE):
        all_para_questions_chunked.append(all_para_questions[i:i + CHUNK_SIZE])

    # send each chunk to the coreference resolution worker
    all_questions_chunked = []
    for chunks in tqdm(pool.imap(coref_worker, all_para_questions_chunked), total=len(all_para_questions_chunked)):
        all_questions_chunked.append(
            chunks
        )

    # collapse the processed chunk list into a single list
    all_questions_chunked = [y for x in all_questions_chunked for y in x]

    instances = []

    for pt, qc, cc, ans, ans_pos in zip(all_para_text, all_questions_chunked, all_conceptual_category, all_answer, all_ans_positions):
        instances.append({
            'paragraph': pt,
            'question': qc,
            'class': cc,
            'answer': ans,
            'answer_position': ans_pos
        })

    print("%d multi-paragraph answers in %s data filtered out" % (partial_answers, corpus))
    print("%d zero length answers in %s data filtered out" % (zero_length_answers, corpus))
    print("%d questions blacklisted in %s data" % (black_listed, corpus))
    print("%d total Q/A pairs %s data" % (len(instances), corpus))

    print("Downsampling questions with a threshold of %d" % DOWNSAMPLE_THRESHOLD)

    counter_qa = Counter([inst['question'] for inst in instances])

    # Downsample the generic questions not captured in the blacklist
    frequent_questions = {}
    for k, v in counter_qa.items():
        if v > DOWNSAMPLE_THRESHOLD:
            frequent_questions[k] = v

    filtered_instances = []

    for inst in instances:
        question = inst['question']
        if question in frequent_questions:
            frequent_questions[question] = frequent_questions[question] - 1
            if frequent_questions[question] <= DOWNSAMPLE_THRESHOLD:
                del frequent_questions[question]
            continue
        filtered_instances.append(inst)

    print("After downsampling, %d Q/A pairs remaining in %s corpus" % (len(filtered_instances), corpus))
    counter_qa = Counter([inst['question'] for inst in filtered_instances])
    print(Counter([v for k, v in counter_qa.items()]))

    with open('data/temp_dataset/instances_corefs_%s.pickle' % corpus, 'wb') as f:
        pickle.dump(filtered_instances, f)
