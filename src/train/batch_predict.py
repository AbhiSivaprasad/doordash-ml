import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import List


def batch_predict(*args, strategy='greedy', **kwargs):
    """Dispatch appropriate prediction function based on strategy"""
    if strategy == 'greedy':
        return batch_predict_greedy(l1_model, l2_models_dict, data_loader)
    else:
        raise ValueError("Invalid batch prediction strategy supplied:", strategy)


def batch_predict_greedy(l1_model: nn.Module,
                         l2_models_dict: Dict[int, nn.Module], 
                         data_loader: DataLoader):
    """Compute batch predictions greedily by maximizing L1, L2 classifiers separately

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    # compute L1 predictions
    l1_predictions = np.array(predict(l1_model, data_loader))

    # compute all L2 predictions
    all_predictions = np.full((len(l1_predictions), len(l2_models_dict)), -1)
    for l1_class_id, model in l2_models_dict.items():
        all_predictions[:, l1_class_id] = predict(model, data_loader)

    if -1 in all_predictions:
        raise Exception("Some predictions missing")
    
    # gather predictions
    return np.take_along_axis(all_predictions, l1_predictions[:, np.newaxis], axis=1)
