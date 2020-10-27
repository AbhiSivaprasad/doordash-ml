import os
import random
import torch
import numpy as np
import torch.nn as nn

from os import walk
from os.path import dirname, join
from argparse import Namespace

from ..models.models import get_model
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, VAL_RESULTS_FILE_NAME


def save_checkpoint(model: nn.Module, args: TrainArgs, val_acc: float, dir_path: str):
    # save model, args
    torch.save({
        "args": Namespace(**args.as_dict()),
        "state_dict": model.state_dict(),
    }, join(dir_path, MODEL_FILE_NAME))

    # save model & tokenizer
    model.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)

    # save accuracy in file
    save_validation_metrics(join(dir_path, VAL_RESULTS_FILE_NAME), val_acc)


def save_validation_metrics(path: str, accuracy: float):
    """Save all validation metrics in a file in path"""
    with open(path, "w") as f:
        f.write(f"Accuracy, {accuracy}")


def read_validaton_metrics(path: str) -> float:
    """Read all validation metrics from file"""
    with open(path, "r") as f:
        _, accuracy = ", ".split(f.readline())

    return float(accuracy)


def load_best_model(path: str):
    """ 
    Find all *.val in subdirectories, they contain validation accuracy of the model in same dir 
    Then load the model the best performing model

    :param path: path of root directory to search for models
    """
    model_results_paths = [join(dirpath, filename) 
                           for dirpath, dirnames, filenames in walk(path) 
                           for filename in [f for f in filenames if f.endswith(".val")]]

    # iterate through paths and track best model
    best_acc = 0
    best_path = None
    for path in model_results_paths:
        acc = read_validation_metrics(path)
        if acc > best:
            best_acc = acc
            best_path = path

    # read model in same directory
    if best_path is not None:
        model_path = join(dirname(best_path), MODEL_FILE_NAME)
        return load_checkpoint(model_path)
    else:
        raise Exception("No model results found")


def set_seed(seed: int):
    """set seed for reproducibility"""
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.device_count() > 0: 
        torch.cuda.manual_seed_all(seed)


class DefaultLogger():
    debug = info = print
