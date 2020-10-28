import os
import random
import torch
import numpy as np
import torch.nn as nn

from os import walk
from os.path import dirname, join
from argparse import Namespace
from transformers import PreTrainedTokenizer

from ..models.models import get_model_class
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, VAL_RESULTS_FILE_NAME, TRAINING_ARGS_FILE_NAME


def save_checkpoint(model: nn.Module, 
                    tokenizer: PreTrainedTokenizer, 
                    args: TrainArgs, 
                    dir_path: str):
    """Save model and training args in dir_path"""
    # List[str] cannot be pickled currently
    args.save(join(dir_path, TRAINING_ARGS_FILE_NAME), skip_unpicklable=True)  

    # save model & tokenizer
    model.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)


def load_checkpoint(dir_path: str):
    """Load model saved in directory dir_path"""
    # read training args
    args = TrainArgs().load(join(dir_path, TRAINING_ARGS_FILE_NAME), 
                            skip_unsettable=True)

    # fetch model and tokenizer from model name in saved training args
    model_cls, tokenizer_cls = get_model_class(args)
    return model_cls.from_pretrained(dir_path), tokenizer_cls.from_pretrained(dir_path)


def save_validation_metrics(dir_path: str, accuracy: float):
    """Save all validation metrics in a file in dir_path"""
    with open(join(dir_path, VAL_RESULTS_FILE_NAME), "w") as f:
        f.write(f"Accuracy, {accuracy}")


def read_validaton_metrics(dir_path: str) -> float:
    """Read all validation metrics from validation file in dir_path"""
    with open(join(dir_path, VAL_RESULTS_FILE_NAME), "r") as f:
        _, accuracy = ", ".split(f.readline())

    return float(accuracy)


def load_best_model(path: str):
    """ 
    Find all *.val in subdirectories, they contain validation accuracy of the model in same dir 
    Then load the model the best performing model

    :param path: path of root directory to search for models
    """
    model_results_dirs = [dirpath
                          for dirpath, dirnames, filenames in walk(path) 
                          for filename in [f for f in filenames if f.endswith(".val")]]

    # iterate through paths and track best model
    best_acc = 0
    best_path = None
    for dir_path in model_results_dirs:
        acc = read_validation_metrics(dir_path)
        if acc > best:
            best_acc = acc
            best_path = path

    # read model in same directory
    if best_path is not None:
        # read model from best_path dir
        model_cls = get_model(args.model_name)
        model.from_pretrained(best_path), tokenizer.from_pretrained(best_path)
        return model, tokenizer
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
