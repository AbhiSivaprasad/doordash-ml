import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import List, Tuple


def batch_predict(*args, strategy='greedy', **kwargs):
    """Dispatch appropriate prediction function based on strategy"""
    if strategy == 'greedy':
        return batch_predict_greedy(l1_model, l2_models_dict, data_loader)
    else:
        raise ValueError("Invalid batch prediction strategy supplied:", strategy)


def batch_predict_greedy(l1_model: nn.Module,
                         l2_models_dict: Dict[int, nn.Module], 
                         data_loader: DataLoader) -> List[Tuple[int]]:
    """Compute batch predictions greedily by maximizing L1, L2 classifiers separately

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    # compute L1 predictions
    l1_predictions = np.array(predict(l1_model, data_loader))
    l2_predictions = np.zeros(len(l1_predictions))

    # compute all L2 predictions
    for l1_class_id, model in l2_models_dict.items():
        l2_predictions[l1_predictions == l1_class_id] = predict(model, data_loader)
    
    # return as a list of tuples
    return list(zip(l1_predictions, l2_predictions))


def batch_predict_complete_search():
    pass


def batch_predict_beam_search():
    pass
