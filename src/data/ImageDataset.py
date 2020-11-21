import torchvision.transforms as transforms
import torch
from PIL import Image, ImageOps, ImageFile
import pathlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Transformer:

    def __init__(self, image_size, pad=False):
        self.image_size = image_size
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.val_trans = transforms.Compose([         
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    # TODO: Add random padding 
    def pad_to_square(im):
    
        desired_size = self.image_size
        old_size = im.size  # old_size[0] is in (width, height) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
    
        # use thumbnail() or resize() method to resize the input image
        # thumbnail is a in-place operation
        # im.thumbnail(new_size, Image.ANTIALIAS)
    
        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
        return new_im


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, folder: str, klass: int, image_size: int, extension: str = "jpg", val: bool = False):
        self._data = pathlib.Path(root) / folder
        self.klass = klass
        self.extension = extension
        if val:
            self.trans =  Transformer(image_size).val_trans
        else:
            self.trans =  Transformer(image_size).train_trans
        # Only calculate once how many files are in this folder
        # Could be passed as argument if you precalculate it somehow
        # e.g. ls | wc -l on Linux
        self._length = sum(1 for entry in os.listdir(str(self._data.resolve())))

    def __len__(self):
        # No need to recalculate this value every time
        return self._length

    def __getitem__(self, index):
        # images always follow [0, n-1], so you access them directly
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            image = Image.open(self._data / "{}.{}".format(str(index), self.extension)).convert('RGBA')

            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
            image = background

            image = self.trans(image)
            assert(image.shape == torch.Size([3, 256, 256]))
            return image, self.klass
        except Exception as e:
            print("failed to load image", self._data, str(index))
            print(e)
            return None


