import torch
import torch.nn as nn
import pandas as pd

from logging import Logger
from tqdm import tqdm, trange
from typing import Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .utils import DefaultLogger
from .evaluate import evaluate_predictions


def train(model: nn.Module
          train_dataloader: DataLoader, 
          valid_dataloader: DataLoader, 
          args: TrainArgs, 
          save_dir: str,
          logger: Logger = None):
    """
    Train model for args.epochs, validate after each epoch, and test best model
    """
    # set seed for reproducibility
    set_seed(args.seed)

    if logger is None:
        logger = DefaultLogger()

    # simple loss function, optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # train model
    best_acc = 0
    best_epoch = 0
    for epoch in trange(args.epochs):
        train_epoch(model, data_loader, optimizer, loss_fn, logger)

        # test on validation set after each epoch
        preds = predict(model, valid_dataloader)
        acc = evaluate_predictions(preds, valid_data["target"], logger)
        debug(f"Validation Accuracy: {val_acc}")

        # if model is better then save
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch

            torch.save({
                "args": args,
                "state_dict": model.state_dict(),
            }, os.path.join(save_dir, MODEL_FILE_NAME))


def train_epoch(model: torch.nn.Module, 
                data_loader: DataLoader, 
                optimizer: Optimizer,
                loss_fn: Callable,
                logger: Logger):
    """
    Train model for a single epoch

    :param data_loader: Torch DataLoader with training batches
    """
    # use custom logger for training
    model.train()
    iter_count, total_loss, total_correct, total_steps = 0
    for batch_iter, data in tqdm(enumerate(data_loader), total=len(data_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        # predict and compute loss
        outputs = model(ids, mask)
        loss =  loss_fn(outputs, targets)
        _, preds = torch.max(outputs.data, dim=1)
        
        # track loss/acc
        total_correct += (preds==targets).sum().item()
        total_loss += loss.item()
        total_steps += targets.size(0)
        if batch_iter % 100 == 0:
            debug((f"Epoch: {epoch}, "
                   f"Iter: {iter_count}, "
                   f"Loss: {loss.item() / total_steps}, "
                   f"Accuracy: {(total_correct * 100) / total_steps}"))
            
            # reset stats
            total_loss, total_correct, total_steps = 0
            
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # track iterations across batches
        iter_count += targets.size(0)

    return iter_count


