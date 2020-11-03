from typing import List, Tuple
import numpy as np

import logging


def evaluate_predictions(preds: np.array, 
                         targets: np.array, 
                         logger: logging.Logger = None) -> float:
    """Compute accuracy given predictions and targets"""
    # Currently measure only acc
    return (preds == targets).sum() / len(targets)


def evaluate_batch_predictions(preds: np.ndarray, targets: np.ndarray, num_l1_classes: int):
    """
    Compute accuracy given predictions and targets
    
    :param preds: 2-d numpy array predicted class ids. 
                  preds[:, 0] contains L1 preds and preds[:, 1] contains L2 preds
    :param targets: 2-d numpy array target class ids. 
                    targets[:, 0] contains L1 target and targets[:, 1] contains L2 target
    """
    scores = preds == targets

    # L1 accuracy and Overall accuracy
    l1_overall_acc = scores[:, 0].mean()
    overall_acc = scores.all(axis=1).mean()

    # L2 accuracy per class
    l1_class_accs = {}  # key = class id, value = acc
    for class_id in range(num_l1_classes):
        l1_class_accs[class_id] = scores[targets[:, 0] == class_id].all(axis=1).mean()

    return overall_acc, l1_overall_acc, l1_class_accs 


def evaluate_lr_precision(results: np.ndarray):
    left_precision = np.zeros(len(results))
    right_precision = np.zeros(len(results))
    denom = np.arange(1, len(results) + 1)

    # calculate cumulative means from left and right
    left_precision = results.cumsum() / denom
    right_precision = (results[::-1].cumsum() / denom)[::-1]

    return np.concatenate((left_precision[:, np.newaxis], right_precision[:, np.newaxis]), axis=1)
