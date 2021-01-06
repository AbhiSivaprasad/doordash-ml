import torch
import torch.nn as nn

from typing import List
from pathlib import Path
from os.path import join

from ..models.hybrid import HybridModel
from .huggingface import HuggingfaceHandler
from .resnet import ResnetHandler


class HybridHandler: 
    MODEL_TYPE = 'hybrid'

    def __init__(self,
                 text_model: nn.Module, 
                 vision_model: nn.Module,
                 text_output_head: nn.Module,
                 vision_output_head: nn.Module,
                 vision_model_name: str,
                 num_classes: int,
                 tokenizer = None,
                 optimizer = None,
                 hybrid_embedding_dim: int = None,
                 hybrid_output_head: nn.Module = None,
                 loss_fn = None,
                 labels: List[str] = None,
                 learning_rate: float = 1e-5,
                 hidden_dim: int = 2048,
                 dropout: float = 0.3):
        self.text_model = text_model
        self.vision_model = vision_model
        self.text_output_head = text_output_head
        self.vision_output_head = vision_output_head
        self.hybrid_output_head = hybrid_output_head
        self.hybrid_embedding_dim = hybrid_embedding_dim
        self.vision_model_name = vision_model_name
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.labels = labels
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # build hybrid model 
        self.hybrid_model = HybridModel(self.vision_model, 
                                        self.text_model, 
                                        self.vision_output_head, 
                                        self.text_output_head, 
                                        self.num_classes, 
                                        self.hybrid_embedding_dim,
                                        self.hybrid_output_head,
                                        self.hidden_dim, 
                                        self.dropout) 

        # initialize optimizer
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.hybrid_model.parameters(), lr=self.learning_rate)

    @classmethod
    def load(cls, dir_path: str):
        """Load model saved in directory dir_path"""
        vision_state = torch.load(join(dir_path, 'vision_model.pt'), 
                                  map_location=lambda storage, loc: storage)
    
        # load vision
        vision_model_name = vision_state['vision_model_name']
        num_classes = vision_state['num_classes']

        vision_model = models.__dict__[vision_model_name](num_classes=num_classes)
        vision_model.load_state_dict(vision_state['model'])        

        # load text model
        text_model = AutoModelForSequenceClassification.from_pretrained(dir_path) 
        tokenizer = AutoTokenizer.from_pretrained(dir_path, do_lower_case=False)

        # load output heads
        vision_output_head = torch.load('vision_output_head.pt')
        text_output_head = torch.load('text_output_head.pt')
        hybrid_output_head = torch.load('hybrid_output_head.pt')

        with open(join(dir_path, "labels.json")) as f:
            labels = json.load(f)

        return HybridHandler(
            text_model, 
            vision_model,
            text_output_head,
            vision_output_head,
            vision_model_name,
            num_classes,
            tokenizer,
            hybrid_output_head=hybrid_output_head,
            labels=labels
        )

    @classmethod
    def load_raw(cls, 
                 vision_dir_path: str, 
                 text_dir_path: str, 
                 learning_rate: float, 
                 hidden_dim: int, 
                 dropout: float):
        # load text model
        text_model = AutoModelForSequenceClassification.from_pretrained(text_dir_path) 
        tokenizer = AutoTokenizer.from_pretrained(text_dir_path, do_lower_case=False)

        # vision model state
        state = torch.load(join(vision_dir_path, 'model.pt'), 
                                map_location=lambda storage, loc: storage)
    
        # load vision model
        num_classes = state['num_classes']
        vision_model_name = state['model_name']
        vision_model = models.__dict__[vision_model_name](num_classes=num_classes)
        vision_model.load_state_dict(state['model'])        

        # load labels (text and vision should have same labels)
        with open(join(dir_path, "labels.json")) as f:
            labels = json.load(f)

        # remove and save the output heads from text model
        text_output_head = nn.Sequential([
            text_model.pre_classifier,
            nn.ReLU(),
            text_model.dropout,
            text_model.classifier
        ])

        # the text model should just output the embedding
        text_model = text_model.distilbert

        # remove and save the output heads from vision model
        vision_output_head = vision_model.fc
        vision_model.fc = lambda x: x  # turn off effect of final fc layer

        # set optimizer and loss_fn
        optimizer = torch.optim.Adam
        loss_fn = nn.CrossEntropyLoss()

        # hybrid embedding dim = vision embedding dim + text embedding dim
        # the inputs to the output heads must have the embedding dimensions
        # text output head is a nn.Sequential so use first layer
        hybrid_embedding_dim = vision_output_head.in_features + text_output_head[0].in_features

        return HybridHandler(
            text_model, 
            vision_model, 
            text_output_head, 
            vision_output_head, 
            vision_model_name, 
            num_classes, 
            tokenizer, 
            optimizer, 
            hybrid_embedding_dim=hybrid_embedding_dim,
            loss_fn=loss_fn, 
            labels=labels, 
            learning_rate=learning_rate, 
            hidden_dim=hidden_dim, 
            dropout=dropout
        )

    def save(self, dir_path: str):
        """Save model and training args in dir_path"""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # only save model itself if using distributed training
        vision_model = (self.vision_model.module 
                        if hasattr(self.vision_model, "module") 
                        else self.vision_model)

        state = {
            'vision_model': self.vision_model.state_dict(),
            'vision_model_name': self.model_name,
            'num_classes': self.num_classes
        }

        torch.save(vision_state, join(dir_path, 'vision_model.pt'))

        # save model & tokenizer
        self.text_model.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)

        # save output heads
        torch.save(self.vision_output_head, 'vision_output_head.pt')
        torch.save(self.text_output_head, 'text_output_head.pt')
        torch.save(self.hybrid_output_head, 'hybrid_output_head.pt')

        # save labels
        with open(join(dir_path, "labels.json"), 'w') as f:
            json.dump(self.labels, f)

        with open(join(dir_path, "master-config.json"), 'w') as f:
            json.dump({
                "model_type": self.MODEL_TYPE
            }, f)
