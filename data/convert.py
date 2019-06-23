import pickle

# for corpus in ['train', 'dev']:
#     print("\nSplit = %s" % corpus)
#     with open('high_low/quac_coqa_%s.pickle' % corpus, 'rb') as f:
#         data = pickle.load(f)

#     for j, instance in enumerate(data):
#         instance.pop('labeler')
#         for para in instance['paragraphs']:
#             for qa in para['qas']:
#                 qa['conceptual_category'] = qa['high_low']
#                 qa['labelling_scheme'] = qa['high_low_mode']

#                 if qa['high_low'] == 'overview':
#                     qa['conceptual_category'] = 'general_concept_completion'
#                 elif qa['high_low'] == 'conceptual':
#                     qa['conceptual_category'] = 'specific_concept_completion'

#                 qa.pop('high_low')
#                 qa.pop('high_low_mode')

#     with open('specificity_qa_dataset/%s.pickle' % corpus, 'wb') as f:
#         pickle.dump(data, f)

for corpus in ['train', 'dev', 'corefs_train', 'corefs_dev']:
    print("\nSplit = %s" % corpus)
    with open('specificity_qa_dataset/instances_%s_temp.pkl' % corpus, 'rb') as f:
        data = pickle.load(f)

    for j, instance in enumerate(data):
        if instance['class'] == 'overview':
            instance['class'] = 'general_concept_completion'
        elif instance['class'] == 'conceptual':
            instance['class'] = 'specific_concept_completion'

    with open('specificity_qa_dataset/instances_%s.pickle' % corpus, 'wb') as f:
        pickle.dump(data, f)
