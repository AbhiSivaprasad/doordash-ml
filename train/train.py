import logging
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ../models/DistilBertClassificationModel import DistilBertClassificationModel

def train(model: DistilBertClassificationModel, 
          data_loader: DataLoader, 
          optimizer: Optimizer
          logging: logging.Logger=None):
    # use custom logger for training
    debug = logging.debug if logging is not None else print

    model.train()
    iter_count, total_loss, total_correct, total_steps = 0
    for batch_iter, data in tqdm(enumerate(data_loader), total=len(data_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        # predict and compute loss
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        _, preds = torch.max(outputs.data, dim=1)
        
        # track loss/acc
        total_correct += compute_acc(preds, targets)
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
