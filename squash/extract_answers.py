import argparse
import json
import pickle
import os
import spacy
import time
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

class EventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        with open("squash/generated_outputs/queue/queue.txt", "r") as f:
            data = f.read().strip()
        if len(data) == 0:
            return
        next_key = data.split("\n")[0]

        # Check whether answer extraction is already complete
        if os.path.exists("squash/generated_outputs/inputs/%s/input.pkl" % next_key):
            return
        else:
            print("Extracting answers for %s" % next_key)

        while not os.path.exists('squash/generated_outputs/inputs/%s/metadata.json' % next_key):
            print("Waiting for squash/generated_outputs/inputs/%s/metadata.json" % next_key)
            time.sleep(1.0)

        with open('squash/generated_outputs/inputs/%s/metadata.json' % next_key, 'r') as f:
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

        with open('squash/generated_outputs/inputs/%s/input.pkl' % next_key, 'wb') as f:
            pickle.dump(instances, f)


if __name__ == "__main__":
    path = "squash/generated_outputs/queue"
    event_handler = EventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
