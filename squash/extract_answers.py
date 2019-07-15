import argparse
import json
import pickle
import spacy

parser = argparse.ArgumentParser()

parser.add_argument('--key', type=str, default=None,
                    help='Input text file to use for extracting answers.')

args = parser.parse_args()

nlp = spacy.load('en_core_web_sm')


def get_answer_spans(para_text):
    para_nlp = nlp(para_text)
    sentences = [(x.text, x.start_char) for x in para_nlp.sents]

    entities = []
    entity_dict = {}
    for x in para_nlp.ents:
        if x.text in entity_dict:
            continue
        entity_dict[x.text] = 1
        entities.append((x.text, x.start_char))

    return sentences, entities

with open('squash/temp/%s/metadata.json' % args.key, 'r') as f:
    data = json.loads(f.read())['input_text'].split('\n')

instances = []

for i, para in enumerate(data):

    sentences, entities = get_answer_spans(para)

    # GENERAL questions from sentences of text
    for sent in sentences:
        instances.append({
            'question': 'what is the answer to life the universe and everything?',
            'paragraph': para,
            'class': 'general',
            'answer': sent[0],
            'answer_position': sent[1],
            'para_index': i,
            'algorithm': 'general_sent'
        })

    # SPECIFIC questions mined from sentences of text
    for sent in sentences:
        instances.append({
            'question': 'what is the answer to life the universe and everything?',
            'paragraph': para,
            'class': 'specific',
            'answer': sent[0],
            'answer_position': sent[1],
            'para_index': i,
            'algorithm': 'specific_sent'
        })

    # SPECIFIC questions with entity answers
    for ent in entities:
        instances.append({
            'question': 'what is the answer to life the universe and everything?',
            'paragraph': para,
            'class': 'specific',
            'answer': ent[0],
            'answer_position': ent[1],
            'para_index': i,
            'algorithm': 'specific_entity'
        })

with open('squash/temp/%s/input.pkl' % args.key, 'wb') as f:
    pickle.dump(instances, f)
