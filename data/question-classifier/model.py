import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.seq2vec_encoders import cnn_encoder

logger = get_logger(__name__)

EPSILON = 1e-13


class ELMoClassifier(nn.Module):
    def __init__(self, config, device):
        super(ELMoClassifier, self).__init__()

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        
        self.elmo = Elmo(options_file, weight_file, 1, dropout=config.dropout)
        self.cnn_encoder = cnn_encoder.CnnEncoder(1024, num_filters=100, output_dim=2)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        questions = [q.split() for q in batch['question']]
        question_ids = batch_to_ids(questions).cuda()
        elmo_vectors = self.elmo(question_ids)
        cnn_vector = self.cnn_encoder(elmo_vectors['elmo_representations'][0], elmo_vectors['mask'])
        loss = self.loss(cnn_vector, batch['class'].cuda())
        preds = torch.argmax(cnn_vector, dim=1)
        softmax = torch.nn.functional.softmax(cnn_vector, dim=1)
        return loss, preds, softmax
