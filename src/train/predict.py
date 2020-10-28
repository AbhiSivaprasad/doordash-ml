import logging
import pandas as pd
import numpy as np

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def predict(model: torch.nn.Module, 
            data_loader: DataLoader,
            device: torch.device):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in data_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            batch_targets = data['targets'].to(device, dtype=torch.long)

            # generate outputs
            logits = model(ids, mask)[0]

            # select predictions
            _, batch_preds = torch.max(logits.data, dim=1)

            # track predictions
            preds.extend(batch_preds.tolist())

    return preds

