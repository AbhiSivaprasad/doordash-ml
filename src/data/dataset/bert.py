import pandas as pd
import torch
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        """
        :param data: Pandas Dataframe containing Dataset. 
                     Column "target" contains int target class. Column "name" contains str item name
        """
        self.len = len(data)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        # TODO: batch tokenize 
        inputs = self.tokenizer(
            str(self.data.Name[index]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True # TODO: measure performance of truncated samples
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        item = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
        }

        targets = (torch.tensor(self.data.target[index], dtype=torch.long) 
                   if "target" in self.data.columns 
                   else None)
        
        return item, targets

    def __len__(self):
        return self.len

    @property
    def targets(self):
        return self.data["target"]
