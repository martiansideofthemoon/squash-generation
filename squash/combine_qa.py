import json

with open('squash/temp/generated_questions.json', 'r') as f:
    question_data = json.loads(f.read())

with open('squash/temp/predictions.json', 'r') as f:
    answer_data = json.loads(f.read())

for para in question_data["data"][0]['paragraphs']:
    for qa in para['qas']:
        qa['predicted_answer'] = answer_data[qa['id']]

with open('squash/temp/final_qa_set.json', 'w') as f:
    f.write(json.dumps(question_data))
