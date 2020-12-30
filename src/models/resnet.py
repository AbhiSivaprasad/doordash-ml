import json
import torch
import torch.nn as nn
import torchvision.models as models

from typing import List
from pathlib import Path
from os.path import join

class ResnetModel: 
    MODEL_TYPE = 'resnet'

    def __init__(self,
                 model: nn.Module,
                 optimizer = None,
                 scheduler = None,
                 loss_fn = None,
                 labels: List[str] = None,
                 num_classes: int = None,
                 model_name: str = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.labels = labels
        self.num_classes = num_classes
        self.model_name = model_name

    @classmethod
    def get_model(cls, 
                  model_name: str,  # "resnet18"
                  labels: List[str],
                  num_classes: int,
                  pretrained: bool = True,
                  lr: float = 0.1,
                  lr_decay: float = 0.1,
                  momentum: float = 0.9,
                  weight_decay: float = 1e-4,
                  lr_step_size: int = 30) -> None:
        # create model
        if pretrained: 
            model = models.__dict__[model_name](pretrained=True, num_classes=1000)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = models.__dict__[model_name](num_classes=num_classes)

        optimizer = torch.optim.SGD(
            model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, lr_decay)

        loss_fn = nn.CrossEntropyLoss()

        return ResnetModel(model, optimizer, scheduler, loss_fn, labels, num_classes, model_name)
    
    def save(self, dir_path: str):
        """
        Store a model with full state
        :param save_dir: path to directory in which to save model files
        """
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # save state
        state = {
            'model': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }

        torch.save(state, join(dir_path, 'model.pt'))

        with open(join(dir_path, "labels.json"), 'w') as f:
            json.dump(self.labels, f)

        with open(join(dir_path, "master-config.json"), 'w') as f:
            json.dump({
                "model_type": self.MODEL_TYPE
            }, f)

    @classmethod
    def load(cls, dir_path: str):
        """Load resnet model saved in directory dir_path"""
        state = torch.load(join(dir_path, 'model.pt'), 
                           map_location=lambda storage, loc: storage)
    
        # create model
        model = models.__dict__[state['model_name']](num_classes=state['num_classes'])
        model.load_state_dict(state['model'])        

        with open(join(dir_path, "labels.json")) as f:
            labels = json.load(f)

        return ResnetModel(model, labels=labels, num_classes=state['num_classes'], model_name=state['model_name'])
