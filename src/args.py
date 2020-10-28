import os
import torch

from datetime import datetime
from typing import List, Optional
from tap import Tap


class TrainArgs(Tap):
    save_dir: str = "output"
    """Directly to save log outputs, model, and results"""
    data_path: str
    """Path to data"""
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
    separate_test_path: str = None
    """Use separate path as test set, test_size will be added to train_size"""
    categories: List[str] = ["L1"]
    """List of category names to build classifiers for. 'L1' signifes L1 classifier. 
    'L2' signifies all L2 classifiers."""
    cuda: bool = True
    """Boolean whether to use GPU when training"""
    gpu: int = 0
    """If cuda = True, specifies id of GPU to use in training"""

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

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)
    
    # validators
    def validate_split_sizes(self):
        # validate train/valid/test split sizes
        if not self.train_size + self.valid_size + self.test_size == 1:
            raise ValueError("train_size, valid_size, test_size must sum to 1")

    def validate_categories(self):
        # 'L2' signfies all L2 categories so if it exists 'L1' can be the only other passed in category
        if 'L2' in self.categories:
            required_len = 1 if 'L1' not in self.categories else 2
            if not len(self.categories) == required_len:
                raise ValueError("Invalid use of 'L2' in categories list")

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        # index save_dir by model name and timestamp
        self.save_dir = os.path.join(self.save_dir, 
                                     self.model_name, 
                                     datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        # validate 
        self.validate_split_sizes()
        self.validate_categories()


class PredictArgs(Tap):
    taxonomy_dir: str
    """Path to directory with taxonomy"""
    models_path: str
    """Path to root models directory. There should be a subdirectory structure according to taxonomy
    e.g. subdirs are named L1 categories"""
    test_path: str
    """Path to test set"""
    max_seq_length: int = 100
    """Max sequence length for BERT models. Longer inputs are truncated"""
    predict_batch_size: int = 32
    """Batch size during model prediction"""
