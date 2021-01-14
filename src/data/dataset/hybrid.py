import pandas as pd
import torch
from torch.utils.data import Dataset
from .bert import BertDataset
from .image import ImageDataset


class HybridDataset(Dataset):
    def __init__(self, 
                 image_dataset: ImageDataset, 
                 text_dataset: BertDataset, 
                 val: bool = False, 
                 preserve_na: bool = False):
        """
        :param data: Pandas Dataframe containing Dataset. 
                     Column "target" contains int target class. Column "name" contains str item name
        """
        self.image_dataset = image_dataset
        self.text_dataset = text_dataset
        self._val = val

        if not preserve_na:
            mask = text_dataset.data["Name"].notna() & image_dataset.data["Image Name"].notna()

            self.image_dataset.data = self.image_dataset.data[mask]
            self.image_dataset.data.reset_index(drop=True, inplace=True)

            self.text_dataset.data = self.text_dataset.data[mask]
            self.text_dataset.data.reset_index(drop=True, inplace=True)

        assert len(self.image_dataset) == len(self.text_dataset)
        if self.image_dataset.targets is not None:
            assert self.image_dataset.targets.equals(self.text_dataset.targets)

        self.targets = self.image_dataset.targets

    def __getitem__(self, index):
        image, image_target = self.image_dataset[index]
        text, text_target = self.text_dataset[index]

        # target variables should be the same
        if image_target != text_target:
            print(f"Image target: {image_target}, Text target: {text_target}")
            raise Exception("Image and text targets don't match")

        target = image_target

        # one input must exist
        assert image is not None or text is not None
            
        # collate can't handle 'None', fix later
        if image is None:
            image = -1

        if text is None:
            text = -1

        if target is None:
            target = -1

        return (text, image), target

    def __len__(self):
        return len(self.image_dataset)
    
    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self.image_dataset.val = value
        self._val = value

