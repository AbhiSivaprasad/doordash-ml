import logging
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Tuple


def predict(model: torch.nn.Module, 
            data_loader: DataLoader,
            device: torch.device,
            return_probs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with model on data from data_loader. Return predictions and confidences"""
    model.eval()
    preds = []
    probs_batches = []
    with torch.no_grad():
        for data in data_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            # generate outputs
            logits = model(ids, mask)[0]

            # compute probabilities
            if return_probs:
                probs_batches.append(F.softmax(logits, dim=1))

            # select predictions
            _, batch_preds = torch.max(logits, dim=1)

            # track predictions
            preds.extend(batch_preds.tolist())
    
    # convert from list to numpy
    preds = np.array(preds)

    # stack probabilities from batches and get max probability across classes as confidence score
    probs = None
    if return_probs:
        probs = torch.cat(probs_batches, dim=0)
        probs = torch.max(probs, dim=1)[0].cpu().numpy()

    return preds, probs

