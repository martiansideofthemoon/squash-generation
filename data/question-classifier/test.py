import os
import time
import torch

from torch.utils.data import DataLoader
from torch import optim

from dataloader import ClassifierDataset, load_weights, save_weights
from model import ELMoClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params


def evaluate(model, dev_dataloader, dev_size):
    with torch.no_grad():
        total_loss = 0
        correct = 0
        for i, batch in enumerate(dev_dataloader):
            loss, preds, _ = model(batch)
            if preds[0].cpu().long() == batch['class'][0].cpu().long():
                correct += 1
            total_loss += loss
        dev_loss = total_loss / dev_size
        dev_accuracy = float(correct) / dev_size
    return dev_loss, dev_accuracy


def train(args, device):
    config = args.config
    lr = config.learning_rate

    dev_data = ClassifierDataset(config, split='dev')

    dev_size = len(dev_data)

    dev_dataloader = DataLoader(
        dev_data,
        batch_size=1,
        num_workers=0,
        drop_last=False
    )

    model = ELMoClassifier(config, device)
    model.cuda()
    params = model.parameters()
    logger.info("Total parameters = %d", tally_parameters(model))

    # If the saving directory has no checkpoints, this function will not do anything
    loaded_steps = load_weights(model, args.train_dir)
    best_performance_file = os.path.join(args.train_dir, 'best_performance.txt')
    if os.path.exists(best_performance_file):
        with open(best_performance_file, 'r') as f:
            best_performance = float(f.read().strip())
    else:
        # arbitrarily high number
        best_performance = 0.0
    # TODO - make this loadable from the disk
    best_steps = 0

    if config.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr)
    elif config.optimizer == 'amsgrad':
        optimizer = optim.Adam(params, lr=lr, amsgrad=True)
    else:
        optimizer = optim.SGD(params, lr=lr)

    for e in range(config.epochs):
        # Training loop
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Forward pass
            loss, preds, _ = model(batch)
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

            if (i + 1) % args.print_every == 0:
                logger.info(
                    "Epoch %d / %d, Batch %d / %d, Loss %.4f",
                    e, config.epochs, i, num_batches, loss
                )
            # This is done to save GPU memory over next loop iteration
            del loss

            if (i + 1) % config.eval_freq == 0 or i == num_batches - 1:
                model.eval()
                # Evaluate dev loss every time with this frequency
                dev_loss, dev_accuracy = evaluate(model, dev_dataloader, dev_size)
                total_steps = loaded_steps + e * num_batches + i + 1

                logger.info(
                    "Dev Loss per example after %d epochs, %d steps = %.4f, Dev Accuracy = %.4f",
                    e, i + 1, dev_loss, dev_accuracy
                )
                if dev_accuracy > best_performance:
                    logger.info("Saving best model")
                    best_steps = total_steps
                    save_weights(model, args.best_dir, total_steps)
                    best_performance = dev_accuracy
                    with open(best_performance_file, 'w') as f:
                        f.write(str(best_performance))
                # we set a patience parameter to decide when to scale the lr
                # decay is performed exponetially
                # if (total_steps - best_steps) > config.patience * config.eval_freq:
                #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config.learning_rate_decay
                #     logger.info("Decayed the learning rate to %.7f", optimizer.param_groups[0]['lr'])
                #     # Setting best steps to total steps to prevent rapid decay in LR
                #     best_steps = total_steps
                # Switching back to train mode
                model.train()

        save_weights(model, args.train_dir, loaded_steps + (e + 1) * num_batches)
