import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import List, Tuple


def batch_predict(*args, strategy='greedy', **kwargs):
    """Dispatch appropriate prediction function based on strategy"""
    if strategy == 'greedy':
        return batch_predict_greedy(l1_model, l2_models_dict, data_loader)
    elif strategy == 'complete':
        return batch_predict_complete_search(l1_model, l2_models_dict, data_loader)
    else:
        raise ValueError("Invalid batch prediction strategy supplied:", strategy)


def batch_predict_greedy(l1_model: nn.Module,
                         l2_models_dict: Dict[int, nn.Module], 
                         data_loader: DataLoader) -> List[Tuple[int]]:
    """Compute batch predictions greedily by maximizing L1, L2 classifiers separately

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    # compute L1 predictions
    l1_preds = np.array(predict(l1_model, data_loader))
    l2_preds = np.zeros(len(l1_preds))

    # compute all L2 predictions
    for l1_class_id, model in l2_models_dict.items():
        l2_preds[l1_preds == l1_class_id] = predict(model, data_loader)
    
    # return as a list of tuples
    return list(zip(l1_preds, l2_preds))


def batch_predict_complete_search(l1_model: nn.Module,
                                  l2_models_dict: Dict[int, nn.Module], 
                                  data_loader: DataLoader): -> List[Tuple[int]]:
    """Compute batch predictions by computing probability of every branch 

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    l1_probs = None

    # get max length from models
    l2_probs = np.full((len(l1_predictions), len(l2_models_dict)), -1)
    
    # multiply every L2 probability with its corresponding L1 probability
    agg_probs = l1_probs * l2_probs

    # compute argmax over batch dimension
    agg_argmax = agg_probs.reshape(agg_probs.shape[0], -1).argmax(axis=1)

    # argmax over flattened L1, L2 ---> separate L1 and L2 preds
    l1_preds, l2_preds = np.unravel_index(agg_argmax, agg_probs[0, :, :].shape) 

    # return as a list of tuples
    return list(zip(l1_preds, l2_preds))


def batch_predict_beam_search(l1_model: nn.Module,
                              l2_models_dict: Dict[int, nn.Module], 
                              data_loader: DataLoader,
                              beam_size: int):
"""More computationally efficient than complete search at the cost of not exploring all paths"""
pass
