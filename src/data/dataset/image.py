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
        # define transform
        transformer = Transformer(image_size)
        self.transform = transformer.val_transform if self.val else transformer.train_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # locate target class
        class_id = self.targets[index] if self.targets is not None else -1

        # will skip corrupted images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # locate image
        image_name = self.data.loc[index, "Image Name"]

        # no image for data point
        if pd.isnull(image_name):
            return -1, class_id

        # get the dir image is stored in
        hash_dir = get_image_hashdir(image_name)

        # path to image
        image_path = join(self._image_dir, hash_dir, image_name)

        try:
            image = self.prepare_image(image_path, self._image_size)
            return image, class_id
        except Exception as e:
            print("Failed to load image", self.data, str(index))
            print(e)
            return -1, class_id

    def prepare_image(self, image_path: str, image_size: int):
        # load image
        image = Image.open(image_path).convert('RGBA')

        # create background and paste image
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        image = background

        # transform image
        image = self.transform(image)
        assert(image.shape == torch.Size([3, 256, 256]))  # change?
        return image

    @property
    def targets(self):
        return self.data["target"] if "target" in self.data else None

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

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
