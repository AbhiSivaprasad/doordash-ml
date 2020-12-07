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


def load_best_model(models_dir: str, wandb_api, args: TrainArgs):
    """ 
    From W&B load the best model for the given model type
    """
    runs = api.runs(path=args.project, filters={
        "config.model_name": args.model_name, 
        "config.category_name": args.category_name,
        "order": "+summary_metrics.loss"
    })

    # get first run


    # model should be the only logged artifact
    run.logged_artifacts

    return load_checkpoint(models_dir) 


def is_best_model(dir_path):
    """
    Load best model of same type and best overall model.
    If trained model is better than either then 
    """
    run.use_artifact(artifact_name).download(models_dir)

    current_best_model = load_model(category_name)
    # if model better than current best:


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

  
def set_seed(seed: int):
    """set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


class DefaultLogger():
    debug = info = print
