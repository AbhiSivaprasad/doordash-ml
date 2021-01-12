import time
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

from ..utils import DefaultLogger, set_seed, move_object_to_device
from ..eval.evaluate import evaluate_predictions
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME
from ..predict.predict import predict


def train(model,
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
    
    # train model
    best_loss = None
    patience = args.patience
    for epoch in trange(args.epochs):
        print(patience)
        # train for an epoch
        train_epoch(model.model, 
                    train_dataloader, 
                    model.optimizer, 
                    model.loss_fn, 
                    args.lr, 
                    device, 
                    logger)

        # test on validation set after each epoch
        valid_preds, valid_probs = predict(model.model, valid_dataloader, device, return_probs=True)
        val_acc = evaluate_predictions(valid_preds, valid_targets.cpu().numpy(), logger)
        val_loss = F.nll_loss(torch.log(valid_probs), valid_targets)

        # if model has a scheduler then adjust learning rates
        if hasattr(model, 'scheduler'):
            model.scheduler.step()

        wandb.log({
            "validation loss": val_loss,
            "validation accuracy": val_acc
        })

        # if model is better then save
        print(f"val loss: {val_loss}, best loss: {best_loss}")
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            
            # save model, args
            model.save(save_dir)

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

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_t, targets = data

        # send all input items to gpu
        input_t = move_object_to_device(input_t, device)

        # targets to gpu
        targets = targets.to(device)
        
        # predict and compute loss
        # logits = model(input_ids=input_t[0], attention_mask=input_t[1])[0]
        logits = model(input_t)

        loss = loss_fn(logits, targets)
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


