import os
import torch

from datetime import datetime
from tempfile import TemporaryDirectory
from typing import List, Optional
from typing_extensions import Literal
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

    wandb_project: str = "main"
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


class TrainArgs(CommonArgs):
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
    category_ids: List[str] = None
    """List of category ids to train models for"""
    all_categories: bool = False
    """If True, run on all categories in taxonomy. taxonomy_artifact_identifier must be specified"""
    data_dir: str
    """Path to directory with data"""
    data_sources: List[str]
    """List of W&B artifact identifiers which constructed data in data_dir. For logging."""

    # W & B args
    train_data_filename: str = "train.csv"
    """File name of train data in dataset artifact"""
    taxonomy_artifact_identifier: str = None
    """W&B identifier of taxonomy artifact if 'all' category ids is selected"""

    # Model args
    model_name: str
    """Name of model to train, format depends on model type"""
    model_source: Literal["huggingface", "wandb"] = "huggingface"
    """Source to pull model from"""
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

    def validate_categories(self):
        # if "all categories" option is set then taxonomy must be passed
        if self.all_categories and self.taxonomy_artifact_identifier is None:
            raise ValueError("To run on all categories, specify taxonomy")

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()
       
        # validators
        self.validate_split_sizes()
        self.validate_categories()


class CommonPredictArgs(CommonArgs):
    max_seq_length: int = 100
    """Max sequence length for BERT models. Longer inputs are truncated"""
    batch_size: int = 32
    """Batch size during model prediction"""
 

class PredictArgs(CommonPredictArgs):
    data_dir: str
    """Path to directory with test.csv file"""
    data_sources: List[str]
    """List of W&B artifact identifiers which constructed data in data_dir. For logging."""
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
