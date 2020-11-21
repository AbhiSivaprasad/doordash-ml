import os
import torch
import time
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import List, Optional
from tap import Tap


class CommonArgs(Tap):
    save_dir: str = None
    """Directory to save outputs"""
    cuda: bool = True
    """Boolean whether to use GPU when training"""
    gpu: int = 0
    """If cuda = True, specifies id of GPU to use in training"""
    _timestamp: str = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
    """Timestampe at command run time. Set in process_args"""

    wandb_project: str = "doordash"
    """Name of W&B project"""

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @property
    def timestamp(self):
        """Get timestamp at command run time"""
        return self._timestamp
    
    def process_args(self):
        super(CommonArgs, self).process_args()

        # Create temporary directory as save directory if not provided
        global temp_dir  # Prevents the temporary directory from being deleted upon function return
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name


class ResnetTrainArgs(CommonArgs):

    # Training args
    group: str = None
    num_classes: int = -1
    image_size: int = 256
    lr: float = 0.01
    epochs: int = 80
    architecture: str = "resnet18"
    workers: int = 32
    batch_size: int = 64
    momentum: float = 0.9
    weight_decay: float = 1e-4

    eval: bool = False

    # Starting models
    pretrained: bool = True
    resume: bool = False
    """Model ID of model to start training from"""
    checkpoint: str = None
    """Whether to pad images to fit dimentions"""
    pad: bool = False
    """Unique model ID"""
    id: str = tr(int(time.time()))

    # Directories
    train_directory: "/home/sarah/training/train/"
    validation_directory: "/home/sarah/training/validation/"

    def process_args(self) -> None:
        super(ResnetTrainArgs, self).process_args()
 
class BertTrainArgs(CommonArgs):
    seed: int = 0
    """Seed for reproducibility"""
    max_seq_length: int = 100
    """Max sequence length for BERT models. Longer inputs are truncated"""
    train_batch_size: int = 16
    """Batch size for model training"""
    predict_batch_size: int = 16
    """Batch size for model prediction"""
    train_size: float = 0.8
    """Size of train split"""
    valid_size: float = 0.1
    """Size of validation split"""
    test_size: float = 0.1
    """Size of test split"""
    taxonomy_path: str = None
    """Path to taxonomy mapping categories to class ids"""
    category_ids: List[str]
    """List of category ids to train models for"""

    # W & B args
    train_data_filename: str = "train.csv"
    """File name of train data in dataset artifact"""

    # Model args
    model_name: str
    """Name of model to train"""
    cls_hidden_dim: int = 768
    """Size of hidden layer in classification head"""
    cls_dropout: float = 0.3
    """Dropout after hidden layer activation in clhahlassification head"""

    # Training args
    epochs: int = 5
    """Number of epochs to train model for"""
    lr: float = 1e-5
    """Learning rate for training"""
    patience: int = 3
    """Number of epochs to wait for better val accuracy before early stopping"""

   # validators
    def validate_split_sizes(self):
        # validate train/valid/test split sizes
        if not self.train_size + self.valid_size + self.test_size == 1:
            raise ValueError("train_size, valid_size, test_size must sum to 1")

    def process_args(self) -> None:
        super(BertTrainArgs, self).process_args()
       
        # validators
        self.validate_split_sizes()


class CommonPredictArgs(CommonArgs):
    max_seq_length: int = 100
    """Max sequence length for BERT models. Longer inputs are truncated"""
    batch_size: int = 32
    """Batch size during model prediction"""
 

class PredictArgs(CommonPredictArgs):
    category_id: str
    """Category id to predict for"""
    autoload_best_model: bool = False
    """Recursively sweep models_dir for the model with highest validation score
    if False then models_dir must directly contain the model"""
    model_artifact_identifier: str
    """W&B identifier for desired model artifact"""


class BatchPredictArgs(CommonPredictArgs):
    taxonomy_dir: str
    """Path to directory with taxonomy"""
    strategy: str = "greedy"
    """Strategy to merge L1, L2 predictions. Options: greedy, complete"""
