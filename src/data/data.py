import pandas as pd
import torch
from torch.utils.data import Dataset


def generate_datasets(data: pd.DataFrame):
    # for now hardcode l2
    datasets = [dataset for dataset in data.groupby('l1')]

    # list of tuples (L1 name, dataset)
    return datasets


class BertDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        # TODO: batch tokenize 
        inputs = self.tokenizer(
            str(self.data.name[index]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True # TODO: measure performance of truncated samples
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.target[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len
