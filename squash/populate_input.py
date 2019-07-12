import argparse
import datetime
import json
import pickle
import os
from random import randint

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="quac",
                    help='Dataset whose dev path we want to use.')

parser.add_argument('--top_p', type=float, default=0.9,
                    help='Hyperparameter for nucleus sampling.')
parser.add_argument('--gen_frac', type=float, default=0.5,
                    help='Percentage of general questions to retain.')
parser.add_argument('--spec_frac', type=float, default=0.8,
                    help='Percentage of specific questions to retain.')

args = parser.parse_args()


def main():
    dataset = args.dataset

    with open('data/specificity_qa_dataset/dev.pickle', 'rb') as f:
        data = pickle.load(f)

    data = [x for x in data if x['dataset'] == dataset]

    instance_num = randint(0, len(data) - 1)
    key = '%s_%d' % (dataset, instance_num)

    while os.path.exists('squash/final/%s.json' % key):
        instance_num = randint(0, len(data) - 1)
        key = '%s_%d' % (dataset, instance_num)

    instance = data[instance_num]

    paras = [x['text'][:10000] for x in instance['paragraphs'][:4]]

    os.mkdir("squash/temp/%s" % key)

    metadata = {
        "input_text": "\n".join(paras),
        "key": key,
        "timestamp": str(datetime.datetime.now()),
        "settings": {
            "top_p": args.top_p,
            "gen_frac": args.gen_frac,
            "spec_frac": args.spec_frac
        }
    }

    with open('squash/temp/%s/metadata.json' % key, 'w') as f:
        f.write(json.dumps(metadata))

    print(key)

if __name__ == "__main__":
    main()
