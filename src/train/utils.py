import os
import random
import torch
import numpy as np
import torch.nn as nn

from argparse import Namespace

from ..models.models import get_model
from ..args import TrainArgs


def save_checkpoint(model: nn.Module, args: TrainArgs, path: str):
    torch.save({
        "args": Namespace(**args.as_dict()),
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: str):
    state = torch.load(path)

    # load stored args
    args = TrainArgs()
    args.from_dict(vars(state['args']), skip_unsettable=True)

    # load model state
    model = get_model(args.model)(args.num_target_classes)
    model.load_state_dict(state['state_dict'])
    model.to(args.device)

    return model


def set_seed(seed: int):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.device_count() > 0: 
        torch.cuda.manual_seed_all(seed)


class DefaultLogger():
    debug = info = print
