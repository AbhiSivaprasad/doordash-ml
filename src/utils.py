import os
import random
import torch
import wandb
import numpy as np
import torch.nn as nn

from os import walk
from os.path import dirname, join
from argparse import Namespace
from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizer

from .args import TrainArgs
from .constants import MODEL_FILE_NAME, VAL_RESULTS_FILE_NAME, TRAINING_ARGS_FILE_NAME


def save_checkpoint(model: nn.Module, 
                    tokenizer: PreTrainedTokenizer, 
                    args: TrainArgs, 
                    dir_path: str):
    """Save model and training args in dir_path"""
    # for data parallel, model is stored under .module
    if hasattr(model, 'module'):
        model = model.module

    # save model & tokenizer
    model.save_pretrained(dir_path)
    tokenizer.save_pretrained(dir_path)


def load_checkpoint(dir_path: str):
    """Load model saved in directory dir_path"""
    return (AutoModelForSequenceClassification.from_pretrained(dir_path), 
            AutoTokenizer.from_pretrained(dir_path, do_lower_case=False))


def upload_checkpoint(run, category_id: str, dir_path: str):
    """Save checkpoint to W&B"""
    # save model & tokenizer as artifact
    artifact = wandb.Artifact(f'model-{category_id}', type='model')
    artifact.add_dir(dir_path)
    run.log_artifact(artifact)


def set_seed(seed: int):
    """set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


class DefaultLogger():
    debug = info = print
