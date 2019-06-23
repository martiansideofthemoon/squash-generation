import argparse
import json
import os
import shutil
import sys

parser = argparse.ArgumentParser()

parser.add_argument("-config_file", "--config_file", default="classifier.yml", type=str, help="Config File")
parser.add_argument("-modify-cfg", "--modify-config", default="{}", type=str, help="Modification to configuration")
parser.add_argument("-data_dir", "--data_dir", default="data/squad/", type=str, help="Training / Test data dir")
parser.add_argument("-train_dir", "--train_dir", default="save", type=str, help="training base dir")
parser.add_argument("-best_dir", "--best_dir", default="save_best", type=str, help="best model base dir")
parser.add_argument("-seed", "--seed", default=1, type=int, help="value of the random seed")
parser.add_argument("-job_id", "--job_id", default="classify_0", type=str, help="Run ID")
parser.add_argument("-load_id", "--load_id", default=None, type=str, help="Run ID to load from. Defaults to job_id")
parser.add_argument("-no-cache", "--no-cache", default=False, action="store_true", help="Use cache or not")
parser.add_argument("-print-every", '--print-every', type=int, default=10, help="Frequency of printing training loss for minibatch")

parser.add_argument(
    "-mode", "--mode", default="train", type=str,
    help="train / test / analysis / generate / classify",
    choices=["train", "test", "analysis", "generate", "classify", "classify_coqa", "classify_final"]
)


def modify_arguments(args):
    base_dir = args.train_dir
    args.train_dir = os.path.join(args.train_dir, args.job_id)
    args.best_dir = os.path.join(args.best_dir, args.job_id)

    if args.load_id is None:
        args.load_dir = os.path.join(base_dir, args.job_id)
    else:
        args.load_dir = os.path.join(base_dir, args.load_id)

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    elif args.no_cache is True and args.mode == 'train':
        shutil.rmtree(args.train_dir)
        os.makedirs(args.train_dir)

    if not os.path.exists(args.best_dir):
        os.makedirs(args.best_dir)
    elif args.no_cache is True and args.mode == 'train':
        shutil.rmtree(args.best_dir)
        os.makedirs(args.best_dir)

    if not os.path.exists(args.load_dir):
        # Error, since we are explicitly asking to load from another directory
        # This directory will get created in the case it is the same as job_id
        print("Error in loading directory")
        sys.exit(0)


def modify_config(args, config):
    new_config = json.loads(args.modify_config)
    for k, v in new_config.items():
        config[k] = v
    return config
