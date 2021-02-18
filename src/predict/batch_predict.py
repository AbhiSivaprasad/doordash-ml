import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from tqdm import tqdm

from .predict import predict
from ..data.taxonomy import TaxonomyNode, Taxonomy


def batch_predict(l1_handler: nn.Module,
                  l2_handler_dict: Dict[int, nn.Module], 
                  data_loader: DataLoader,
                  device: torch.device,
                  strategy: str = 'greedy', 
                  return_probs: bool = False,
                  **kwargs):
    """Dispatch appropriate prediction function based on strategy"""
    if strategy == 'greedy':
        return batch_predict_greedy(l1_model, l2_models_dict, data_loader, device)
    elif strategy == 'complete':
        return batch_predict_complete_search(l1_handler, l2_handler_dict, data_loader, device)
    else:
        raise ValueError("Invalid batch prediction strategy supplied:", strategy)


def batch_predict_greedy(l1_model: nn.Module,
                         l2_models_dict: Dict[int, nn.Module], 
                         data_loader: DataLoader,
                         device: torch.device,
                         topk: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute batch predictions greedily by maximizing L1, L2 classifiers separately

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    # compute L1 predictions
    l1_model.to(device)
    l1_preds, l1_probs = predict(l1_model, data_loader, device, return_probs=True)

    topk_tensor = torch.topk(l1_probs, topk)
    l1_topk_classes = topk_tensor.indices.cpu().numpy()
    l1_topk_confidences = topk_tensor.values.cpu().numpy()

    # track L2 predictions and probabilities
    l2_preds = np.zeros((len(l1_preds), topk, topk), dtype=int)
    l2_probs = np.zeros((len(l1_preds), topk, topk))

    # compute all L2 predictions
    for l1_class_id, model in tqdm(l2_models_dict.items()):
        model.to(device)

        l2_class_preds, l2_class_probs = predict(model, data_loader, device, return_probs=True)

        # get l2 preds
        topk_tensor = torch.topk(l2_class_probs, topk)
        l2_topk_classes = topk_tensor.indices.cpu().numpy()
        l2_topk_confidences = topk_tensor.values.cpu().numpy()

        # for each l1 class, add topk l2 predictions
        for i in range(topk):
            mask = (l1[:, i] == l1_class_id)
            l2_probs[mask, i, :] = l2_topk_confidences[mask, :]
            l2_preds[mask, i, :] = l2_topk_classes[mask, :]

        # move model back to save gpu memory
        model.to(torch.device('cpu'))
    
    # broadcast L1 * L2 confidence
    l2_agg_confidences = l1_topk_confidences * l2_probs

    # unwrap confidences and preds to choose top k
    l2_agg_confidences = l2_agg_confidences.reshape(-1, topk * topk) 
    l2_preds = l2_preds.reshape(-1, topk * topk) 

    # find top k 
    topk_tensor = torch.topk(l2_agg_confidences, topk)
    topk_inds = topk_tensor.indices.cpu().numpy()
    topk_confidences = topk_tensor.values.cpu().numpy()

    # index into l2_preds to find classes
    topk_classes = np.take(l2_preds, topk_inds, axis=1)  # FIX

    # preds, L1 confidence, L1 & L2 confidence
    return topk_classes, l1_topk_confidences, topk_confidences


def batch_predict_complete_search(l1_handler: nn.Module,
                                  l2_handlers_dict: Dict[int, nn.Module], 
                                  data_loader: DataLoader,
                                  device: torch.device,
                                  topk: int = 3) -> List[Tuple[int]]:
    """Compute batch predictions by computing probability of every branch 

    :param l2_models_dict: key = int L1 class id, value = corresponding L2 model
    """
    l1_handler.model.to(device)
    _, l1_probs = predict(l1_handler.model, data_loader, device, return_probs=True)

    # get max number of l2 categories
    max_l2_categories = max([handler.num_classes for handler in l2_handlers_dict.values()])

    # create ndarray to save l2 model probs
    l2_probs = np.zeros((l1_probs.shape[0], l1_probs.shape[1], max_l2_categories))
    
    for category_id, handler in tqdm(l2_handlers_dict.items()):
        handler.model.to(device)

        l1_class_id = l1_handler.labels.index(category_id)
        _, l2_class_probs = predict(handler.model, data_loader, device, return_probs=True)
        
        # place computed probs in a slice of dims 0, 2
        l2_probs[:, l1_class_id, :l2_class_probs.shape[1]] = l2_class_probs.cpu().numpy()

        # move back to cpu to keep gpu memory low
        handler.model.to(torch.device('cpu'))

    # multiply every L2 probability with its corresponding L1 probability
    agg_probs = l1_probs[:, :, np.newaxis].cpu().numpy() * l2_probs

    # compute top k over batch dimension
    unwrapped_agg_probs = agg_probs.reshape(agg_probs.shape[0], -1)

    topk_tensor = torch.topk(torch.from_numpy(unwrapped_agg_probs), topk)
    topk_inds = topk_tensor.indices.cpu().numpy()
    topk_confidences = topk_tensor.values.cpu().numpy()

    # argmax over flattened L1, L2 ---> separate L1 and L2 preds
    l1_preds, l2_preds = np.unravel_index(topk_inds, agg_probs[0, :, :].shape) 

    # rework preds
    all_preds = []
    for l1_topk, l2_topk in zip(l1_preds, l2_preds):
        all_preds.append(list(zip(l1_topk, l2_topk)))
    
    # return as a list of tuples
    return all_preds, topk_confidences.tolist()


def batch_predict_beam_search(l1_model: nn.Module,
                              l2_models_dict: Dict[str, nn.Module], 
                              data_loader: DataLoader,
                              data_selector: List[bool],
                              tags: List[Dict[str, List[Tuple[str, float]]]],
                              device: torch.device,
                              topk: int = 3,
                              depth: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """More computationally efficient than complete search at the cost of not exploring all paths"""
    category_id = node.category_id
    model = models[category_id]
    
    # move model to gpu
    model.to(device)

    # predict 
    class_preds, class_probs = predict(model, data_loader, device, return_probs=True)
    
    # top k probabilities
    topk_tensor = torch.topk(class_probs, topk)
    topk_classes = topk_tensor.indices.cpu().numpy()
    topk_confidences = topk_tensor.values.cpu().numpy()

    # add to labels
    label_name = f"L{depth}"
    for i in range(len(data_selector)):
        # only add label to selected data points
        if data_selector[i]:
            topk_class_names = [model_labels[class_id] for class_id in topk_classes[i, :]]
            topk_confidences = list(topk_confidences[i, :])
            tags[label_name] = list(zip(topk_class_names, topk_confidences))

    # conserve gpu memory
    model.to(torch.device('cpu'))
