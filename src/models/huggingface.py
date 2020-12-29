import json
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from pathlib import Path
from os.path import join


class HuggingfaceModel: 
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 optimizer = None,
                 loss_fn = None,
                 labels: List[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.labels = labels

    @classmethod
    def get_model(cls, 
                  pretrained_model_name_or_path: str,
                  labels: List[str],
                  num_classes: int, 
                  lr: float) -> None:
        # load huggingface model & tokenizer
        model, tokenizer = cls._get_model(
            num_classes, pretrained_model_name_or_path)    

        # Adam optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        # simple loss function, optimizer
        loss_fn = nn.CrossEntropyLoss()

        return HuggingfaceModel(model, tokenizer, optimizer, loss_fn, labels)

    def save(self, dir_path: str):
        """Save model and training args in dir_path"""
        # for data parallel, model is stored under .module
        if hasattr(self.model, 'module'):
            model = self.model.module

        # save model & tokenizer
        self.model.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)

        # save labels
        with open(join(dir_path, "labels.json"), 'w') as f:
            json.dump(self.labels, f)

    @classmethod
    def load(cls, dir_path: str):
        """Load model saved in directory dir_path"""
        model = AutoModelForSequenceClassification.from_pretrained(dir_path) 
        tokenizer = AutoTokenizer.from_pretrained(dir_path, do_lower_case=False)

        with open(join(dir_path, "labels.json")) as f:
            labels = json.load(f)

        return HuggingfaceModel(model, tokenizer, labels=labels)

    def _get_model(num_labels: int, pretrained_model_name_or_path: str):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            pad_token="[PAD]"
        )

        if pretrained_model_name_or_path == 'gpt2-medium':
            config.pad_token_id = tokenizer.get_vocab()["[PAD]"]

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
        )
        
        if pretrained_model_name_or_path == 'gpt2-medium':
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer
