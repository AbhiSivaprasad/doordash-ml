import logging
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Tuple

from ..utils import move_object_to_device


def predict(model: torch.nn.Module, 
            data_loader: DataLoader,
            device: torch.device,
            return_probs: bool = False) -> Tuple[np.array, torch.Tensor]:
    """Predict with model on data from data_loader. Return predictions and confidences"""
    model.eval()
    preds = []
    probs_batches = []
    with torch.no_grad():
        for input_t, _ in data_loader:
            # send all input items to gpu
            input_t = move_object_to_device(input_t, device) 

            # generate outputs
            logits = model(input_t)

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
    #     probs = torch.max(probs, dim=1)[0].cpu().numpy()

    return preds, probs

