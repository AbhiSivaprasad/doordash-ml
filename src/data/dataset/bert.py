import pandas as pd
import torch
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int, preserve_na: bool = False):
        """
        :param data: Pandas Dataframe containing Dataset. 
                     Column "target" contains int target class. Column "name" contains str item name
        """
        if not preserve_na:
            self.data = data[data["Name"].notna()]
            self.data.reset_index(drop=True, inplace=True)
        else:
            self.data = data

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preserve_na = preserve_na

       
    def __getitem__(self, index):
        name = self.data.Name[index]

        # no name provided for item
        if pd.isnull(name):
            return None, None

        inputs = self.tokenizer(
            str(name),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True # TODO: measure performance of truncated samples
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        item = [torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)]

        targets = (torch.tensor(self.data.target[index], dtype=torch.long) 
                   if "target" in self.data.columns 
                   else -1)
        
        return item, targets

    def __len__(self):
        return len(self.data)

    @property
    def targets(self):
        return self.data["target"]
