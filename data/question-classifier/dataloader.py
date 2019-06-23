import glob
import os
import sys
import pickle
import torch

from torch.utils.data import Dataset
from utils.logger import get_logger

from collections import Counter

logger = get_logger(__name__)

PAD_token = 0
SOS_token = 1
EOS_token = 2


def get_checkpoints(base_path):
    checkpoints = glob.glob(os.path.join(base_path, 'checkpoint-*'))
    checkpoints.sort(key=lambda x: int(x[x.rfind('-') + 1:]))
    return checkpoints


def save_weights(model, base_path, steps, keep=2):
    path = os.path.join(base_path, 'checkpoint-%d' % steps)
    torch.save(model.state_dict(), path)
    # Remove older checkpoint files
    checkpoints = get_checkpoints(base_path)
    for checkpoint in checkpoints[:-1 * keep]:
        os.remove(checkpoint)
    if os.path.exists(path) is False:
        logger.info("Bug in code, you deleted the file you just saved!")
        sys.exit(0)
    logger.info("Saved model checkpoint %s", path)


def load_weights(model, base_path, steps=None):
    if steps is None:
        # Load the latest checkpoint
        checkpoints = get_checkpoints(base_path)
        if len(checkpoints) == 0:
            logger.info('loading 0 completed steps')
            return 0
        else:
            latest_ckpt = checkpoints[-1]
            model.load_state_dict(torch.load(latest_ckpt))
            latest_steps = int(latest_ckpt[latest_ckpt.rfind('-') + 1:])
            logger.info('loading %d completed steps', latest_steps)
            return latest_steps
    else:
        path = os.path.join(base_path, 'checkpoint-%d', steps)
        if os.path.exists(path) is False:
            logger.info('Path %s does not exists, loaded 0 completed steps', path)
            return 0
        else:
            logger.info('loading %d completed steps', steps)
            return steps


def process(question):
    if question.strip()[-1] != '?':
        question += '?'
    while len(question.split()) < 5:
        question += ' ?'
    return question.replace('?', ' ?').strip()


class ClassifierDataset(Dataset):
    """Generic wrapper to load chain data, the para level data can be reused"""
    def __init__(self, config, split='train'):
        with open('%s_classifier.pickle' % split, 'rb') as f:
            self.data = pickle.load(f)
        for i, instance in enumerate(self.data):
            instance['question'] = process(instance['question'])
            instance['class'] = 0 if instance['label'] == 'overview' else 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'question': self.data[idx]['question'],
            'label': self.data[idx]['label'],
            'class': self.data[idx]['class']
        }

    def process(self, question):
        if question.strip()[-1] != '?':
            question += '?'
        while len(question.split()) < 5:
            question += ' ?'
        return question.replace('?', ' ?').strip()
