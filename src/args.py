import os
import torch

from datetime import datetime
from typing import List, Optional
from tap import Tap


class TrainArgs(Tap):
    model: str
    """Name of model to train"""
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
    categories: List[str] = ["All"]
    """List of category names to build classifiers for. 'All' signifes L1 classifier"""
    cuda: bool = True
    """Boolean whether to use GPU when training"""
    gpu: int = 0
    """If cuda = True, specifies id of GPU to use in training"""

    # Training args
    epochs: int = 5
    """Number of epochs to train model for"""
    lr: float = 1e-5
    """Learning rate for training"""

    # non-user args
    _num_target_classes = None
    """Number of target classes. Not passed by user, auto set during execution"""

    @property
    def num_target_classes(self) -> Optional[int]:
        """
        Number of target classes.

        Computed during execution so will return None if accessed before value is set
        """
        return self._num_target_classes 

    @num_target_classes.setter
    def num_target_classes(self, num_classes):
        self._num_target_classes = num_classes

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        # validate train/valid/test split sizes
        if not self.train_size + self.valid_size + self.test_size == 1:
            raise ValueError("train_size, valid_size, test_size must sum to 1")

        # index save_dir by model name and timestamp
        self.save_dir = os.path.join(self.save_dir, 
                                     self.model, 
                                     datetime.now().strftime("%Y%m%d-%H%M%S"))
    
