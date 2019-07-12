import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--key', type=str, default=None,
                    help='Input text file to use for extracting answers.')

args = parser.parse_args()

with open('squash/temp/%s/generated_questions.json' % args.key, 'r') as f:
    question_data = json.loads(f.read())

with open('squash/temp/%s/predictions.json' % args.key, 'r') as f:
    answer_data = json.loads(f.read())

for para in question_data["data"][0]['paragraphs']:
    for qa in para['qas']:
        qa['predicted_answer'] = answer_data[qa['id']]

with open('squash/temp/%s/final_qa_set.json' % args.key, 'w') as f:
    f.write(json.dumps(question_data))
