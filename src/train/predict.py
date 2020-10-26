import logging
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def predict(model: torch.nn.Module, data_loader: DataLoader):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in tqdm(testing_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            batch_targets = data['targets'].to(device, dtype=torch.long)

            # generate outputs
            outputs = model(ids, mask).squeeze()

            # select predictions
            _, batch_preds = torch.max(outputs.data, dim=1)

            # track predictions
            preds.extend(batch_preds.tolist())

    return preds
