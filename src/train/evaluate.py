from typing import List

import logging

def evaluate_predictions(preds: List[int], 
                         targets: List[int], 
                         logger: logging.Logger = None) -> float:
    preds = np.array(preds) 
    targets = np.array(targets)
   
    # Currently measure only acc
    return (preds == targets).sum() / len(targets)

