from typing import List, Tuple
import numpy as np

import logging


def evaluate_predictions(preds: List[int], 
                         targets: List[int], 
                         logger: logging.Logger = None) -> float:
    """Compute accuracy given predictions and targets"""
    preds = np.array(preds) 
    targets = np.array(targets)

    # Currently measure only acc
    return (preds == targets).sum() / len(targets)


def evaluate_batch_predictions(preds: List[Tuple[int]], targets: List[Tuple[int]], num_classes: int):
    """Compute accuracy given predictions and targets
    
    :param preds: List of Tuples with predicted class ids. Format: [(L1, L2), (L1, L2), ...]
    """
    preds = np.array(preds)
    targets = np.array(targets)

    scores = preds == targets

    # L1 accuracy and Overall accuracy
    l1_acc = scores[:, 0].mean()
    total_acc = scores.all(axis=1).mean()

    # L2 accuracy per class
    l2_accs = {}  # key = class id, value = acc
    for class_id in range(num_classes):
        l2_accs[class_id] = scores[targets == class_id].all(axis=1).mean()

    return total_acc, l1_acc, l2_accs
