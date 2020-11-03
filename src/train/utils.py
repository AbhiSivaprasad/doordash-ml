import os
import random
import torch
import numpy as np
import torch.nn as nn

from os import walk
from os.path import dirname, join
from argparse import Namespace
from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizer

from ..models.models import get_model_class
from ..args import TrainArgs
from ..constants import MODEL_FILE_NAME, VAL_RESULTS_FILE_NAME, TRAINING_ARGS_FILE_NAME


def save_checkpoint(model: nn.Module, 
                    tokenizer: PreTrainedTokenizer, 
                    args: TrainArgs, 
                    dir_path: str):
    """Save model and training args in dir_path"""
    # save model & tokenizer
    model.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)


def load_checkpoint(dir_path: str):
    """Load model saved in directory dir_path"""
    return (AutoModelForSequenceClassification.from_pretrained(dir_path), 
            AutoTokenizer.from_pretrained(dir_path, do_lower_case=False))


def save_validation_metrics(dir_path: str, accuracy: float, loss: float):
    """Save all validation metrics in a file in dir_path"""
    with open(join(dir_path, VAL_RESULTS_FILE_NAME), "w") as f:
        f.write(f"Accuracy, {accuracy}\n")
        f.write(f"Loss, {loss}\n")


def read_validation_metrics(dir_path: str) -> float:
    """Read all validation metrics from validation file in dir_path"""
    with open(join(dir_path, VAL_RESULTS_FILE_NAME), "r") as f:
        _, accuracy = f.readline().split(", ")
        _, loss = f.readline().split(", ")

    return float(accuracy), float(loss)


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
    best_loss = None
    best_path = None
    for dir_path in model_results_dirs:
        _, loss = read_validation_metrics(dir_path)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_path = dir_path

    # read model in same directory
    if best_path is not None:
        # read model from best_path dir
        return load_checkpoint(best_path) 
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
