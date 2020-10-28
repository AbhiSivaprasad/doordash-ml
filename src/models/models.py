import torch
import torch.nn as nn

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

from ..args import TrainArgs


def get_model(num_labels: int, args: TrainArgs):
    if args.model_name == 'distilbert':
        config = DistilBertConfig(num_labels=num_labels, 
                                  seq_classif_dropout=args.cls_dropout, 
                                  cls_hidden_dim=args.cls_hidden_dim)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased') 
        model = DistilBertForSequenceClassification(config)
        return model, tokenizer
    else:
        raise ValueError("Invalid model type:", args.model_name)


def get_model_class(args: TrainArgs):
    if args.model_name == 'distilbert':
        tokenizer = DistilBertTokenizer
        model = DistilBertForSequenceClassification
        return model, tokenizer
    else:
        raise ValueError("Invalid model type:", args.model_name)
