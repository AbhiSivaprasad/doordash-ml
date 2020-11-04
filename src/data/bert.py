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
            # truncation=True # TODO: measure performance of truncated samples
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        item = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }

        if "target" in self.data.columns:
            item["targets"] = torch.tensor(self.data.target[index], dtype=torch.long) 
        
        return item

    def __len__(self):
        return self.len

    @property
    def targets(self):
        return self.data["target"]
