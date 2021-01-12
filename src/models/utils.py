import json

from os.path import join
from typing import List
from .handlers.resnet import ResnetHandler
from .handlers.huggingface import HuggingfaceHandler
from .handlers.hybrid import HybridHandler
from ..args import TrainArgs


def get_hyperparams(model_type: str):
    if model_type == ResnetHandler.MODEL_TYPE:
        return [
            'pretrained',
            'momentum',
            'weight_decay',
            'lr',
            'lr_decay',
            'lr_step_size'
        ]
    elif model_type == HuggingfaceHandler.MODEL_TYPE:
        return [
            'cls_dropout',
            'cls_hidden_dim',
            'max_seq_length',
            'lr'
        ]
    elif model_type == HybridHandler.MODEL_TYPE:
        return [
            'lr',
            'cls_hidden_dim',
            'cls_dropout'
        ]
    else:
        raise ValueError("Invalid model type")


def get_model_handler(args: TrainArgs, 
                      labels: List[str], 
                      num_classes: int, 
                      vision_model_path: str = None, 
                      text_model_path: str = None):
    """
    Build initial model given hyperparameters

    :param model_path: If finetuning then path to existing model
    """
    model_type = args.model_type

    if model_type == ResnetHandler.MODEL_TYPE:
        return ResnetHandler.load_raw(args.model_name, 
                                      labels, 
                                      num_classes, 
                                      args.pretrained, 
                                      args.lr, 
                                      args.lr_decay,
                                      args.momentum, 
                                      args.weight_decay,
                                      args.lr_step_size)
    elif model_type == HuggingfaceHandler.MODEL_TYPE:
        return HuggingfaceHandler.load_raw(args.model_name, 
                                           labels, 
                                           num_classes,
                                           args.lr,
                                           text_model_path) 
    elif model_type == HybridHandler.MODEL_TYPE:
        return HybridHandler.load_raw(vision_model_path,
                                      text_model_path,
                                      args.lr,
                                      args.cls_hidden_dim,
                                      args.cls_dropout)
    else:
        raise ValueError("Invalid model type")


def load_model_handler(dir_path: str):
    with open(join(dir_path, "master-config.json")) as f:
         config = json.load(f)

    # extract model type
    model_type = config["model_type"]

    if model_type == ResnetHandler.MODEL_TYPE:
        return ResnetHandler.load(dir_path)
    elif model_type == HuggingfaceHandler.MODEL_TYPE:
        return HuggingfaceHandler.load(dir_path)
    elif model_type == HybridHandler.MODEL_TYPE:
        return HybridHandler.load(dir_path)
    else:
        raise ValueError("Invalid model type")
 
