from typing import List
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
