import random
import torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.device_count() > 0: 
        torch.cuda.manual_seed_all(seed)


class DefaultLogger():
    debug = info = print
