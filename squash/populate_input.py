import pickle
import os
from random import randint

dataset = 'quac'

with open('data/high_low/quac_coqa_dev.pickle', 'rb') as f:
    data = pickle.load(f)

data = [x for x in data if x['dataset'] == dataset]

instance_num = randint(0, len(data) - 1)
key = '%s_%d' % (dataset, instance_num)

while os.path.exists('squash/final/%s.json' % key):
    instance_num = randint(0, len(data) - 1)
    key = '%s_%d' % (dataset, instance_num)

print("Choosing sequence number %d from dev set" % instance_num)
instance = data[instance_num]

paras = [x['text'] for x in instance['paragraphs'][:5]]

with open('squash/temp/input.txt', 'w') as f:
    f.write('\n'.join(paras))

with open('squash/temp/instance.txt', 'w') as f:
    f.write('%s_%d' % (dataset, instance_num))
