import pickle
import unicodedata
import re
import sys

from collections import Counter

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

for corpus in ['train', 'dev']:
    print("\nSplit = %s" % corpus)
    with open('/mnt/nfs/work1/miyyer/kalpesh/projects/squash-generation/data/high_low/quac_coqa_%s.pickle' % corpus, 'rb') as f:
        data = pickle.load(f)

    instances = []

    partial_answers = 0
    zero_length_answers = 0
    black_listed = 0

    for j, instance in enumerate(data):
        for para in instance['paragraphs']:

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

                instances.append({
                    'paragraph': para['text'].strip(),
                    'question': qa['question'].strip(),
                    'class': qa['high_low'],
                    'answer': qa['answer'].strip()
                })

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

    with open('instances_%s.pkl' % corpus, 'wb') as f:
        pickle.dump(filtered_instances, f)
