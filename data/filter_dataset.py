import pickle
import unicodedata
import re

from collections import Counter

from blacklist import black_list, black_list_patterns

DOWNSAMPLE_THRESHOLD = 10


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

for corpus in ['train', 'dev']:
    print("\nSplit = %s" % corpus)
    with open('data/specificity_qa_dataset/%s.pickle' % corpus, 'rb') as f:
        data = pickle.load(f)

    instances = []

    partial_answers = 0
    zero_length_answers = 0
    black_listed = 0

    for j, instance in enumerate(data):
        for para in instance['paragraphs']:

            for qa in para['qas']:
                # Remove answers spanning multiple paragraphs
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

                instances.append({
                    'paragraph': para['text'],
                    'question': qa['question'].strip(),
                    'class': qa['conceptual_category'],
                    'answer': qa['answer'],
                    'answer_position': qa['local_ans_position']
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

    with open('data/temp_dataset/instances_%s.pickle' % corpus, 'wb') as f:
        pickle.dump(filtered_instances, f)
