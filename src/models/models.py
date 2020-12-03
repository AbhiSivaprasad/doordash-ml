import wandb
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tempfile import TemporaryDirectory

from ..args import TrainArgs
from ..utils import load_checkpoint


def get_wandb_model(wandb_api, model_artifact_identifier, save_dir=None):
    # if no save directory passed in then create a temp dir
    temp_dir = None
    if save_dir is None:
        temp_dir = TemporaryDirectory()
        save_dir = temp_dir.name

    # download model artifact from W&B
    wandb_api.artifact(model_artifact_identifier).download(save_dir)

    # load checkpoint from downloaded artifact
    return load_checkpoint(save_dir)
     

def get_huggingface_model(num_labels: int, args: TrainArgs):
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        pad_token="[PAD]"
    )

    if args.model_name == 'gpt2-medium':
        config.pad_token_id = tokenizer.get_vocab()["[PAD]"]

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )
    
    if args.model_name == 'gpt2-medium':
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
