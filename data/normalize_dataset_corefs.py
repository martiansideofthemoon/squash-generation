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


def get_word(text, location):
    """Identify the word at this location"""
    reverse_text = text[::-1]
    reverse_location = len(text) - location - 1
    try:
        start_index = len(text) - reverse_text.index(' ', reverse_location)
    except:
        start_index = 0
    try:
        end_index = text.index(' ', location)
    except:
        end_index = len(text)
    return text[start_index:end_index]


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def process(ip):
    output = ip.replace('"', '').replace('\'', '').strip()
    return output


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


for corpus in ['train']:
    print("\nSplit = %s" % corpus)
    with open('/mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/data/high_low/quac_coqa_%s.pickle' % corpus, 'rb') as f:
        data = pickle.load(f)

    instances = []

    partial_answers = 0
    zero_length_answers = 0
    black_listed = 0
    inst_num = 0

    all_para_text = []
    all_para_questions = []
    all_high_low = []
    all_answer = []

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

                if qa['question'].lower() in black_list:
                    black_listed += 1
                    continue

                if qa['high_low'] == 'verification':
                    continue

                pattern_match_found = False
                for pattern in black_list_patterns:
                    if len(re.findall(pattern, qa['question'].lower())) > 0:
                        pattern_match_found = True

                if pattern_match_found is True:
                    black_listed += 1
                    continue

                current_para_text = ' '.join([x['text'] for x in instance['paragraphs'][:para_num + 1]])
                para_question = current_para_text.strip() + " ~ " + qa['question'].strip()

                all_para_text.append(para['text'].strip())
                all_para_questions.append(para_question)
                all_high_low.append(qa['high_low'])
                all_answer.append(qa['answer'].strip())

    num_processes = 48
    chunk_size = 5
    pool = Pool(processes=num_processes)

    # chunk questions into blocks of chunk_size
    all_para_questions_chunked = []
    for i in range(0, len(all_para_questions), chunk_size):
        all_para_questions_chunked.append(all_para_questions[i:i + chunk_size])

    all_questions_chunked = []
    for chunks in tqdm(pool.imap(coref_worker, all_para_questions_chunked), total=len(all_para_questions_chunked)):
        all_questions_chunked.append(
            chunks
        )

    all_questions_chunked = [y for x in all_questions_chunked for y in x]

    for para_text, question_chunked, high_low, answer in zip(all_para_text, all_questions_chunked, all_high_low, all_answer):
        instances.append({
            'paragraph': para_text,
            'question': question_chunked,
            'class': high_low,
            'answer': answer
        })

    # inst_num += 1

    # if inst_num % 1000 == 0:
    #     with open('instances_corefs_%s_temp.pkl' % corpus, 'wb') as f:
    #         pickle.dump(instances, f)

    print("%d multi-paragraph answers in %s data filtered out" % (partial_answers, corpus))
    print("%d zero length answers in %s data filtered out" % (zero_length_answers, corpus))
    print("%d questions blacklisted in %s data" % (black_listed, corpus))
    print("%d total Q/A pairs %s data" % (len(instances), corpus))

    print("Downsampling questions with a threshold of %d" % DOWNSAMPLE_THRESHOLD)

    counter_qa = Counter([inst['question'] for inst in instances])

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

    with open('instances_corefs_%s.pkl' % corpus, 'wb') as f:
        pickle.dump(filtered_instances, f)
