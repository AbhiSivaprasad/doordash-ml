import pandas as pd
import torch
from torch.utils.data import Dataset


class HybridDataset(Dataset):
    def __init__(self, image_dataset: ImageDataset, text_dataset: TextDataset):
        """
        :param data: Pandas Dataframe containing Dataset. 
                     Column "target" contains int target class. Column "name" contains str item name
        """
        self.image_dataset = image_dataset
        self.text_dataset = text_dataset        

        assert len(self.image_dataset) == len(self.text_dataset)
        assert self.image_dataset.targets.equals(self.text_dataset.targets)

        self.target = self.image_dataset.targets

    def __getitem__(self, index):
        image, image_target = self.image_dataset[index]
        text, text_target = self.text_dataset[index]

        # target variables should be the same
        assert image_target == text_target
        target = image_target

        return (text, image), target

    def __len__(self):
        return len(self.image_dataset)

    @property
    def targets(self):
        return self.data["target"]
