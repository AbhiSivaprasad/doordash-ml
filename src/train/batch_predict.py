import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from tqdm import tqdm

from .predict import predict


def batch_predict(l1_model: nn.Module,
                  l2_models_dict: Dict[int, nn.Module], 
                  data_loader: DataLoader,
                  device: torch.device, 
                  strategy: str = 'greedy', 
                  return_probs: bool = False,
                  **kwargs):
    """Dispatch appropriate prediction function based on strategy"""
    if strategy == 'greedy':
        return batch_predict_greedy(l1_model, l2_models_dict, data_loader, device)
    elif strategy == 'complete':
        return batch_predict_complete_search(l1_model, l2_models_dict, data_loader, device)
    else:
        raise ValueError("Invalid batch prediction strategy supplied:", strategy)


def batch_predict_greedy(l1_model: nn.Module,
                         l2_models_dict: Dict[int, nn.Module], 
                         data_loader: DataLoader,
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute batch predictions greedily by maximizing L1, L2 classifiers separately

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    # compute L1 predictions
    l1_model.to(device)
    l1_preds, l1_probs = predict(l1_model, data_loader, device, return_probs=True)

    # find probability of predicted class as confidence score
    l1_probs = torch.max(l1_probs, dim=1)[0].cpu().numpy()

    # track L2 predictions and probabilities
    l2_preds = np.zeros(len(l1_preds), dtype=int)
    l2_probs = np.zeros(len(l1_preds))

    # compute all L2 predictions
    for l1_class_id, model in tqdm(l2_models_dict.items()):
        model.to(device)

        l2_class_preds, l2_class_probs = predict(model, data_loader, device, return_probs=True)

        # find probability of predicted class as confidence score
        l2_class_probs = torch.max(l2_class_probs, dim=1)[0].cpu().numpy()

        # select only data predicted as current category
        mask = (l1_preds == l1_class_id)

        # select the l2_preds, and l2_probs for the current category 
        l2_preds[mask] = l2_class_preds[mask]
        l2_probs[mask] = l2_class_probs[mask]  

        # move model back to save gpu memory
        model.to(torch.device('cpu'))
    
    # join predictions. (batch, 2)
    preds = np.concatenate((l1_preds[:, np.newaxis], l2_preds[:, np.newaxis]), axis=1)

    # preds, L1 confidence, L1 & L2 confidence
    return preds, l1_probs, l1_probs * l2_probs


def batch_predict_complete_search(l1_model: nn.Module,
                                  l2_models_dict: Dict[int, nn.Module], 
                                  data_loader: DataLoader,
                                  device: torch.device) -> List[Tuple[int]]:
    """Compute batch predictions by computing probability of every branch 

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    l1_model.to(device)
    _, l1_probs = predict(l1_model, data_loader, device, return_probs=True)

    # get max number of l2 categories
    max_l2_categories = max([model.config.num_labels for model in l2_models_dict.values()])

    # create ndarray to save l2 model probs
    l2_probs = np.zeros((l1_probs.shape[0], l1_probs.shape[1], max_l2_categories))
    
    for l1_class_id, model in tqdm(l2_models_dict.items()):
        model.to(device)
        _, l2_class_probs = predict(model, data_loader, device, return_probs=True)

        # place computed probs in a slice of dims 0, 2
        l2_probs[:, l1_class_id, :l2_class_probs.shape[1]] = l2_class_probs

        # move back to cpu to keep gpu memory low
        model.to(torch.device('cpu'))
    
    # multiply every L2 probability with its corresponding L1 probability
    agg_probs = l1_probs[:, :, np.newaxis] * l2_probs

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
