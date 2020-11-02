import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from logging import Logger
from tqdm import tqdm, trange
from typing import Callable, List
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from .utils import DefaultLogger, set_seed, save_checkpoint
from .evaluate import evaluate_predictions
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME
from .predict import predict
from .evaluate import evaluate_predictions


def train(model: nn.Module,
          tokenizer: PreTrainedTokenizer,
          train_dataloader: DataLoader, 
          valid_dataloader: DataLoader, 
          valid_targets: np.array,
          args: TrainArgs, 
          save_dir: str,
          device: torch.device,
          logger: Logger = None):
    """
    Train model for args.epochs, validate after each epoch, and test best model
    """
    # set seed for reproducibility
    set_seed(args.seed)

    # default logger prints
    logger = DefaultLogger() if logger is None else logger

    # simple loss function, optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # train model
    best_acc = 0
    best_epoch = 0
    for epoch in trange(args.epochs):
        logger.debug("Epoch:", epoch)

        # train for an epoch
        train_epoch(model, train_dataloader, optimizer, loss_fn, device, logger)

        # test on validation set after each epoch
        valid_preds = predict(model, valid_dataloader, device)[0]
        val_acc = evaluate_predictions(valid_preds, valid_targets, logger)
        logger.debug(f"Validation Accuracy: {val_acc}")

        # if model is better then save
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            
            # save model, args
            save_checkpoint(model, tokenizer, args, save_dir)


def train_epoch(model: torch.nn.Module, 
                data_loader: DataLoader, 
                optimizer: Optimizer,
                loss_fn: Callable,
                device: torch.device,
                logger: Logger):
    """
    Train model for a single epoch

    :param data_loader: Torch DataLoader with training batches
    """
    # use custom logger for training
    model.train()
    iter_count = total_loss = total_correct = total_steps = 0
    for batch_iter, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        
        # predict and compute loss
        logits = model(ids, mask)[0]
        loss = F.cross_entropy(logits, targets)
        _, preds = torch.max(logits.data, dim=1)
        
        # track loss/acc
        total_correct += (preds==targets).sum().item()
        total_loss += loss.item()
        total_steps += targets.size(0)
        if batch_iter % 100 == 0:
            logger.debug((f"Iter: {iter_count}, "
                          f"Loss: {loss.item() / total_steps}, "
                          f"Accuracy: {(total_correct * 100) / total_steps}"))
            
            # reset stats
            total_loss = total_correct = total_steps = 0
            
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # track iterations across batches
        iter_count += targets.size(0)

    return iter_count



