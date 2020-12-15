import os
import pathlib
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFile
   

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 image_dir: str, 
                 image_size: int, 
                 extension: str = "jpg", 
                 val: bool = False):
        self._data = data
        self._image_dir = image_dir
        self._extension = extension

        # image transform
        self.transform = (Transformer(image_size).val_transform 
                          if val 
                          else Transformer(image_size).train_transform)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # what does this actually do?

        # locate image
        image_name = self._data.loc[index, "Image Name"]
        image_path = join(self._image_dir, f"{image_name}.{self._extension}")

        # locate target class
        class_id = self._data.loc[index, "Target"]

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
