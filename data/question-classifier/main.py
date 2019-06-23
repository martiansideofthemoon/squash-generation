import yaml

import numpy as np
import random
import torch

from munch import Munch

from arguments import modify_arguments, modify_config, parser

import train
import analysis

from utils.logger import get_logger

logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parser.parse_args()
    modify_arguments(args)

    # setting random seeds
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.config_file, 'r') as stream:
        config = yaml.load(stream)
        args.config = Munch(modify_config(args, config))
    logger.info(args)

    if args.mode == 'train':
        train.train(args, device)
    elif args.mode == 'test':
        pass
    elif args.mode == 'analysis':
        analysis.analyze(args, device)
    elif args.mode == 'generate':
        pass
    elif args.mode == 'classify':
        analysis.classify(args, device)
    elif args.mode == 'classify_coqa':
        analysis.classify_coqa(args, device)
    elif args.mode == 'classify_final':
        analysis.classify_final(args, device)

if __name__ == '__main__':
    main()
