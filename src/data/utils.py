import pandas as pd

from ..args import TrainArgs
from ..models.handlers.huggingface import HuggingfaceHandler
from ..models.handlers.resnet import ResnetHandler
from ..models.handlers.hybrid import HybridHandler
from .dataset.bert import BertDataset
from .dataset.image import ImageDataset
from .dataset.hybrid import HybridDataset


def get_dataset(data: pd.DataFrame, args: TrainArgs, handler = None):
    if args.model_type == HuggingfaceHandler.MODEL_TYPE:
        assert handler is not None
        assert handler.tokenizer is not None
        return BertDataset(data, handler.tokenizer, args.max_seq_length)
    elif args.model_type == ResnetHandler.MODEL_TYPE:
        return ImageDataset(data, args.image_dir, args.image_size)
    elif args.model_type == HybridHandler.MODEL_TYPE:
        assert handler is not None
        assert handler.tokenizer is not None
        text_dataset = BertDataset(data, handler.tokenizer, args.max_seq_length, preserve_na=True)
        image_dataset = ImageDataset(data, args.image_dir, args.image_size, preserve_na=True)
        return HybridDataset(image_dataset, text_dataset)
