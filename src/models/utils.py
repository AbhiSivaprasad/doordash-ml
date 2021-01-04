import json

from os.path import join
from typing import List
from ..models.resnet import ResnetModel
from ..models.huggingface import HuggingfaceModel
from ..args import TrainArgs


def get_hyperparams(model_type: str):
    if model_type == 'resnet':
        return [
            'pretrained',
            'momentum',
            'weight_decay',
            'lr',
            'lr_decay',
            'lr_step_size'
        ]
    elif model_type == 'huggingface':
        return [
            'cls_dropout',
            'cls_hidden_dim',
            'max_seq_length',
            'lr'
        ]
    else:
        raise ValueError("Invalid model type")


def get_model(args: TrainArgs, labels: List[str], num_classes: int, model_path: str = None):
    """
    Build initial model given hyperparameters

    :param model_path: If finetuning then path to existing model
    """
    model_type = args.model_type

    if model_type == 'resnet':
        return ResnetModel.get_model(args.model_name, 
                                     labels, 
                                     num_classes, 
                                     args.pretrained, 
                                     args.lr, 
                                     args.lr_decay,
                                     args.momentum, 
                                     args.weight_decay,
                                     args.lr_step_size)
    elif model_type == 'huggingface':
        return HuggingfaceModel.get_model(args.model_name, 
                                          labels, 
                                          num_classes,
                                          args.lr,
                                          model_path) 
    else:
        raise ValueError("Invalid model type")


def load_model(dir_path: str):
    with open(join(dir_path, "master-config.json")) as f:
         config = json.load(f)

    # extract model type
    model_type = config["model_type"]

    if model_type == 'resnet':
        return ResnetModel.load(dir_path)
    elif model_type == 'huggingface':
        return HuggingfaceModel.load(dir_path)
    else:
        raise ValueError("Invalid model type")
 
