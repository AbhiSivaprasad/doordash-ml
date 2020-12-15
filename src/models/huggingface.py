import torch
import torchvision.models as models

from pathlib import Path
from os.path import join

class HuggingfaceModel: 
    def __init__(self, 
                 num_classes: int, 
                 pretrained_model_name_or_path: str,
                 learning_rate: float = 0.1) -> None:
        self.tokenizer = 
        self.model = 
   
    def save(dir_path: str):
        """Save model and training args in dir_path"""
        # for data parallel, model is stored under .module
        if hasattr(self.model, 'module'):
            model = self.model.module

        # save model & tokenizer
        model.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)

    def load(dir_path: str):
        """Load model saved in directory dir_path"""
        model = AutoModelForSequenceClassification.from_pretrained(dir_path) 
        tokenizer = AutoTokenizer.from_pretrained(dir_path, do_lower_case=False)
        
        self
