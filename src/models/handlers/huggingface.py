import json
import torch
import torch.nn as nn
import torchvision.models as models

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from pathlib import Path
from os.path import join


class HuggingfaceHandler: 
    MODEL_TYPE = 'huggingface'

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

        # standardize forward pass by saving __call__ as __callsave__
        # __call__ is reassigned to process inputs then route to __callsave__
        #if '__callsave__' not in dir(self.model):
        #    self.model.__class__.__callsave__ = self.model.__call__
        #    self.model.__class__.__call__ = self.standardized_forward

    @classmethod
    def download_model(cls, 
                       model_name: str,
                       labels: List[str],
                       num_classes: int, 
                       lr: float) -> None:
        # If no model path, then model name is a huggingface name to be downloaded
        model_identifier = model_path if model_path is not None else model_name

        # load huggingface model & tokenizer
        model, tokenizer = cls._get_model(
            num_classes, model_identifier)    

        # Adam optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        # simple loss function, optimizer
        loss_fn = nn.CrossEntropyLoss()

        return HuggingfaceHandler(model, tokenizer, optimizer, loss_fn, labels)

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

        with open(join(dir_path, "master-config.json"), 'w') as f:
            json.dump({
                "model_type": self.MODEL_TYPE
            }, f)

    @classmethod
    def load(cls, dir_path: str):
        """Load model saved in directory dir_path"""
        model = AutoModelForSequenceClassification.from_pretrained(dir_path) 
        tokenizer = AutoTokenizer.from_pretrained(dir_path, do_lower_case=False)

        with open(join(dir_path, "labels.json")) as f:
            labels = json.load(f)

        return HuggingfaceHandler(model, tokenizer, labels=labels)

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

    def standardized_forward(self, input_list):
        input_ids, attention_mask = input_list
        return self.model.__callsave__(input_ids=input_ids, attention_mask=attention_mask)[0]

