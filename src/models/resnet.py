import torch
import torchvision.models as models

from pathlib import Path
from os.path import join

class ResnetModel: 
    def __init__(self, 
                 num_classes: int, 
                 lr: float = 0.1, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-4, 
                 architecture: str = "resnet18", 
                 pretrained: bool = False) -> None:
        self.num_classes = num_classes
        self.architecture = architecture
 
        # create model
        if pretrained: 
            model = models.__dict__[arch](pretrained=True, num_classes=1000)
            model.fc = torch.nn.Linear(512, num_classes)
        else:
            model = models.__dict__[arch](num_classes=num_classes)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    
    def save(self, save_dir: str):
        """
        Store a model with full state
        :param save_dir: path to directory in which to save model files
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # save state
        state = {
            'state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(state, join(save_dir, 'model.pt'))

        with open(join(model_dir, "labels.json"), 'w') as f:
            json.dump(labels, f)

    def load(self, checkpoint_file: str, device: torch.device):
        if cpu:
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(checkpoint_file)
    
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        with open(join(category_dir, "labels.json")) as f:
            labels = json.load(f)
