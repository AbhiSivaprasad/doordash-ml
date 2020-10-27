import torch.nn as nn

from torch.utils.data import DataLoader
from typing import List


def batch_predict(models: List[nn.Module], data_loader: DataLoader):
    for model in models:
        predictions = predict(model, data_loader)
    
