import os
import torch

from datetime import datetime
from time import time
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
    max_seq_length: int = 256
    """Max sequence length for BERT models. Longer inputs are truncated"""
    train_batch_size: int = 32
    """Batch size for model training"""
    predict_batch_size: int = 32
    """Batch size for model prediction"""
    train_size: float = 0.8
    """Size of train split"""
    valid_size: float = 0.1
    """Size of validation split"""
    test_size: float = 0.1
    """Size of test split"""
    category_ids: List[str] = None
    """List of category ids to train models for"""
    all_categories: bool = False
    """If True, run on all categories in taxonomy. taxonomy_artifact_identifier must be specified"""
    train_dir: str
    """Path to directory with data"""
    test_dir: str = None
    """
    Path to directory with test data. Dir contains one dir per category and a 'test.csv' in each
    Automatically adds test_size to train_size
    """
    train_data_sources: List[str]
    """List of W&B artifact identifiers which constructed data in train_dir. For logging."""
    test_data_sources: List[str]
    """List of W&B artifact identifiers which constructed data in test_dir. For logging."""
    batch_id: str = str(int(time()))
    """Unified id for all runs launched by one script. Defaults to timestamp"""

    # W & B args
    taxonomy_artifact_identifier: str = None
    """W&B identifier of taxonomy artifact if 'all' category ids is selected"""

    # Model args
    model_name: str
    """Name of model to train, format depends on model type"""
    vision_model_dir: str = None
    """Directory with one subdirectory per category containing a model. 
    Used if loading model from dir for vision or hybrid"""
    text_model_dir: str = None
    """Directory with one subdirectory per category containing a model. 
    Used if loading model from dir for text or hybrid"""
    model_type: Literal["huggingface", "resnet", "hybrid"] = "huggingface"
    """Type of model to train"""
    # """Source to pull model from"""
    cls_hidden_dim: int = 768
    """Size of hidden layer in classification head"""
    cls_dropout: float = 0.3
    """Dropout after hidden layer activation in classification head"""

    # Resnet Model args
    pretrained: bool = True
    """Whether to use pretrained weights"""
    momentum: float = 0.9
    """Momentum for training optimizer"""
    weight_decay: float = 1e-4
    """Weight decay for training optimizer"""
    image_dir: str = "local/images"
    """Path to directory with images"""
    image_size: int = 256
    """Size of image inputs"""
    lr_decay: float = 0.1
    """Every lr_step_size epochs, decay learning rate"""
    lr_step_size: float = 20
    """Every lr_step_size epochs, decay learning rate"""

    # Training args
    epochs: int = 10
    """Number of epochs to train model for"""
    lr: float = 1e-5  # default for bert finetuning
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

        # if test dir passed, then train data is split into (train, val) not (train, val, test)
        # test size is added to train size by default
        if self.test_dir is not None:
            self.train_size += self.test_size
            self.test_size = 0


class CommonPredictArgs(CommonArgs):
    max_seq_length: int = 100
    """Max sequence length for BERT models. Longer inputs are truncated"""
    predict_batch_size: int = 32
    """Batch size during model prediction"""
    image_dir: str = None
    """Path to directory with images if using image model"""
    image_size: int = 256
    """Size of image inputs"""
    model_type: Literal["huggingface", "resnet", "hybrid"] = "huggingface"
    """Type of model to train"""
    model_dir: str
    """Path to directory with models. Contains subdirectories named after category id with one model each"""
    taxonomy: str = None
    """W&B identifier of taxonomy artifact if 'all' category ids is selected"""
 
class PredictArgs(CommonPredictArgs):
    data_dir: str
    """Path to test directory"""
    eval_datasets: List[str]
    """List of W&B artifact identifiers which constructed data in data_dir. For logging."""
    category_ids: List[str] = []
    """Category ids to predict for"""
    all_categories: bool = False
    """If True, run on all categories in taxonomy. taxonomy_artifact_identifier must be specified"""


class BatchPredictArgs(CommonPredictArgs):
    test_path: str
    """Path to csv to test on"""
    write_path: str
    """Path to save csv results"""
    strategy: str = "greedy"
    """Strategy to merge L1, L2 predictions. Options: greedy, complete"""

