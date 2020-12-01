import os
import wandb
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

from ..utils import DefaultLogger, set_seed, save_checkpoint
from ..eval.evaluate import evaluate_predictions
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME
from ..predict.predict import predict


def train(model: nn.Module,
          tokenizer: PreTrainedTokenizer,
          train_dataloader: DataLoader, 
          valid_dataloader: DataLoader, 
          valid_targets: torch.tensor,
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
    best_loss = None
    patience = args.patience
    for epoch in trange(args.epochs):
        logger.debug("Epoch:", epoch)

        # train for an epoch
        train_epoch(model, train_dataloader, optimizer, loss_fn, args.lr, device, logger)

        # test on validation set after each epoch
        valid_preds, valid_probs = predict(model, valid_dataloader, device, return_probs=True)
        val_acc = evaluate_predictions(valid_preds, valid_targets.cpu().numpy(), logger)
        val_loss = F.nll_loss(torch.log(valid_probs), valid_targets)

        wandb.log({
            "validation loss": val_loss,
            "validation accuracy": val_acc
        }, commit=False)

        # if model is better then save
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            
            # save model, args
            save_checkpoint(model, tokenizer, args, save_dir)

            # reset patience
            patience = args.patience
        else:
            # val loss has not improved
            patience -= 1

        # early stopping condition
        if patience == 0:
            break


def train_epoch(model: torch.nn.Module, 
                data_loader: DataLoader, 
                optimizer: Optimizer,
                loss_fn: Callable,
                learning_rate: float,
                device: torch.device,
                logger: Logger):
    """
    Train model for a single epoch

    :param data_loader: Torch DataLoader with training batches
    """
    # use custom logger for training
    model.train()
    for batch_iter, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        
        # predict and compute loss
        logits = model(input_ids=ids, attention_mask=mask)[0]
        loss = F.cross_entropy(logits, targets)
        _, preds = torch.max(logits.data, dim=1)
          
        # log to W&B, autotracks iter
        wandb.log({
            "train loss": loss.item(),
            "train accuracy": (preds==targets).sum().item() / targets.size(0),
        })
 
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
