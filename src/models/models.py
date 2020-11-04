import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from ..args import TrainArgs


def get_model(num_labels: int, args: TrainArgs):
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
