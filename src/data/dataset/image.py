import os
import hashlib
import pathlib
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFile
from os.path import join
from ..image_utils import get_image_hashdir
   

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 image_dir: str, 
                 image_size: int, 
                 val: bool = False,
                 preserve_na: bool = False):
        self._image_dir = image_dir
        self._image_size = image_size

        # remove rows without images
        if not preserve_na:
            self.data = data[data["Image Name"].notna()]
            self.data.reset_index(drop=True, inplace=True)
        else:
            self.data = data

        # image transform
        self._val = val
        self.transform = self.get_transform(self.val)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # locate target class
        class_id = self.data.loc[index, "target"]

        # will skip corrupted images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # locate image
        image_name = self.data.loc[index, "Image Name"]

        if pd.isnull(image_name):
            return None, class_id

        # get the dir image is stored in
        hash_dir = get_image_hashdir(image_name)

        # path to image
        image_path = join(self._image_dir, hash_dir, image_name)

        try:
            # load image
            image = Image.open(image_path).convert('RGBA')

            # create background and paste image
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
            image = background

            # transform image
            image = self.transform(image)
            assert(image.shape == torch.Size([3, 256, 256]))  # change?
            return image, class_id
        except Exception as e:
            print("failed to load image", self.data, str(index))
            print(e)
            return None

    def get_transform(self, val: bool):
        return (Transformer(self._image_size).val_transform 
                if val 
                else Transformer(self._image_size).train_transform)

    @property
    def targets(self):
        return self.data["target"]

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

        # reset transform
        self.transform = self.get_transform(self.val)


class Transformer:
    def __init__(self, image_size, pad=False):
        self.image_size = image_size
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.val_transform = transforms.Compose([         
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
