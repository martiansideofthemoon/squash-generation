import csv
import os
import pickle
import torch

from torch.utils.data import DataLoader

from dataloader import ClassifierDataset, load_weights, process
from model import ELMoClassifier
from utils.logger import get_logger

from question_rules import labeller

logger = get_logger(__name__)


def classify_final(args, device):
    args.config.batch_size = 1
    config = args.config

    model = ELMoClassifier(config, device)
    model.cuda()
    # If the saving directory has no checkpoints, this function will not do anything
    load_weights(model, args.best_dir)

    correct_class = {'gold': 0, 'gen': 0}
    questions_so_far = {'gold': {}, 'gen': {}}
    total = {'gold': 0, 'gen': 0}

    with torch.no_grad():
        model.eval()
        with open('doc2qa/final/final_crowd/results/question_more_relevant.csv', 'r') as f:
            data = csv.reader(f)
            for i, row in enumerate(data):
                if row[8][:3] == 'Zen':
                    print("yolo")
                    continue
                if i == 0:
                    continue
                if row[2] == 'golden':
                    continue
                for ques, tag in zip([row[12], row[13]], [row[16], row[17]]):
                    if ques in questions_so_far[row[9]]:
                        continue
                    total[row[9]] += 1
                    auto_label = labeller(ques)
                    if auto_label == 'none':
                        _, preds, _ = model({
                            'question': [process(ques)],
                            'class': torch.LongTensor([0])
                        })
                        if preds.item() == 0:
                            auto_label = 'overview'
                        else:
                            auto_label = 'conceptual'
                    questions_so_far[row[9]][ques] = 1
                    if auto_label in ['overview', 'causal', 'instrumental', 'judgemental'] and tag == 'high':
                        print(auto_label)
                        correct_class[row[9]] += 1
                    elif auto_label == 'conceptual' and tag == 'low':
                        correct_class[row[9]] += 1
            print("Gold correct class = %d / %d" % (correct_class['gold'], total['gold']))
            print("Gen correct class = %d / %d" % (correct_class['gen'], total['gen']))



def classify(args, device):
    args.config.batch_size = 1
    config = args.config

    model = ELMoClassifier(config, device)
    model.cuda()
    # If the saving directory has no checkpoints, this function will not do anything
    load_weights(model, args.best_dir)

    with open('data/specificity_qa_dataset/dev.pickle', 'rb') as f:
        data = pickle.load(f)

    with torch.no_grad():
        model.eval()
        for i, instance in enumerate(data):
            if instance['dataset'] == 'quac' or instance['dataset'] == 'coqa':
                continue
            if i % 1 == 0:
                print("%d / %d" % (i, len(data)))
            for para in instance['paragraphs']:
                for qa in para['qas']:
                    if qa['high_low_mode'] == 'idk':
                        if len(qa['question'].strip()) == 0:
                            qa['high_low'] = 'overview'
                        else:
                            _, preds, _ = model({
                                'question': [process(qa['question'])],
                                'class': torch.LongTensor([0])
                            })
                            qa['high_low_mode'] = 'classifier'
                            qa['high_low'] = 'overview' if preds.item() == 0 else 'conceptual'

    with open('data/specificity_qa_dataset/dev.pickle', 'wb') as f:
        pickle.dump(data, f)


def classify_coqa(args, device):
    args.config.batch_size = 1
    config = args.config

    model = ELMoClassifier(config, device)
    model.cuda()
    # If the saving directory has no checkpoints, this function will not do anything
    load_weights(model, args.best_dir)

    with open('data/specificity_qa_dataset/dev.pickle', 'rb') as f:
        data = pickle.load(f)

    with torch.no_grad():
        model.eval()
        correct_hand = 0
        incorrect_hand = 0
        correct_rule = 0
        incorrect_rule = 0
        class_map = ['overview', 'conceptual']
        for i, instance in enumerate(data):
            if instance['dataset'] == 'quac':
                continue
            if i % 100 == 0:
                print("%d / %d" % (i, len(data)))
            for para in instance['paragraphs']:
                for qa in para['qas']:
                    if qa['high_low_mode'] == 'rules':
                        continue
                    if qa['high_low'] == 'overview' or qa['high_low'] == 'conceptual':
                        _, preds, _ = model({
                            'question': [process(qa['question'])],
                            'class': torch.LongTensor([0])
                        })
                        if qa['high_low_mode'] == 'hand' and class_map[preds.item()] == qa['high_low']:
                            correct_hand += 1
                        elif qa['high_low_mode'] == 'rules' and class_map[preds.item()] == qa['high_low']:
                            correct_rule += 1
                        elif qa['high_low_mode'] == 'hand' and class_map[preds.item()] != qa['high_low']:
                            incorrect_hand += 1
                        elif qa['high_low_mode'] == 'rules' and class_map[preds.item()] != qa['high_low']:
                            incorrect_rule += 1
        print("%d / %d correct for hand" % (correct_hand, correct_hand + incorrect_hand))
        print("%d / %d correct for rules" % (correct_rule, correct_rule + incorrect_rule))
        print("%d / %d total correct" % (correct_rule + correct_hand, correct_rule + incorrect_rule + correct_hand + incorrect_hand))


def analyze(args, device):
    args.config.batch_size = 1
    config = args.config

    dev_data = ClassifierDataset(config, split='dev')

    dev_dataloader = DataLoader(
        dev_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = ELMoClassifier(config, device)
    model.cuda()
    # If the saving directory has no checkpoints, this function will not do anything
    load_weights(model, args.best_dir)

    with torch.no_grad():
        model.eval()
        num_correct = 0
        total_loss = 0.0
        for i, batch in enumerate(dev_dataloader):
            loss, preds, softmax = model(batch)
            correct = "WRONG"
            if preds[0].cpu().long() == batch['class'][0].cpu().long():
                num_correct += 1
                correct = "CORRECT"
            if correct == 'WRONG':
                logger.info("%s\t%s\t%s" % (correct, softmax, batch['question'][0]))
            total_loss += loss
        logger.info("Accuracy = %.4f" % (num_correct / len(dev_data)))
