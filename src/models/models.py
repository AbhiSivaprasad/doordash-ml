import torch
import torch.nn as nn
from transformers import DistilBertModel

# Simple dropout + dense layer on top of DistilBERT
class DistilBertClassificationModel(torch.nn.Module):
    def __init__(self, output_size):
        super(DistilBertClassificationModel, self).__init__()
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, output_size)

    def forward(self, input_ids, attention_mask):
        output_1 = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def get_model(model_name: str):
    if model_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased') 
        return DistilBertClassificationModel, tokenizer
    else:
        raise ValueError("Invalid model type:", model_name)
