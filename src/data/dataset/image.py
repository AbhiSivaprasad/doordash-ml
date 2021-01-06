import os
import hashlib
import pathlib
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFile
from os.path import join
   

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 image_dir: str, 
                 image_size: int, 
                 val: bool = False,
                 preserve_na: bool = False):
        self._image_dir = image_dir

        # remove rows without images
        if not preserve_na:
            self._data = data[data["Image Name"].notna()]
            self._data.reset_index(drop=True, inplace=True)

        # image transform
        self.transform = (Transformer(image_size).val_transform 
                          if val 
                          else Transformer(image_size).train_transform)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # will skip corrupted images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # locate image
        image_name = self._data.loc[index, "Image Name"]

        if pd.isnull(image_name):
            return None, None

        # get the dir image is stored in
        hash_dir = self.get_hash_dir(image_name)

        # path to image
        image_path = join(self._image_dir, hash_dir, image_name)

        # locate target class
        class_id = self._data.loc[index, "target"]

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
            print("failed to load image", self._data, str(index))
            print(e)
            return None

    def get_hash_dir(self, image_name: str):
        stripped_image_name = ".".join(image_name.split(".")[:-1])

        # if the stripped name is "" then extension not found
        assert stripped_image_name != ""

        # first two chars of hash
        return hashlib.sha1(stripped_image_name.encode('utf-8')).hexdigest()[:2]

    @property
    def targets(self):
        return self._data["target"]


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
