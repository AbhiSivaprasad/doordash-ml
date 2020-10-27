import pandas as pd
import numpy as np

from typing import List


def generate_datasets(train_data: pd.DataFrame, 
                      valid_data: pd.DataFrame, 
                      test_data: pd.DataFrame,
                      categories: List[str]):
    """
    Given a dataset, generate subsets for each category
    """
    datasets = [train_data, valid_data, test_data]

    # for now hardcode l1, l2 
    l1_info = {"name": "All", "n_classes": len(set(train_data['L1']))}
    l1_datasets = (
        [(l1_info, [dataset for dataset in datasets])]
        if "All" in categories else []
    )

    l2_datasets = []
    for category in categories:
        n_l2_classes = len(set(train_data.loc[train_data['L1'] == category, 'L2']))
        l2_info = {"name": category, "n_classes": n_l2_classes}

        l2_datasets.append((l2_info, [dataset[dataset['L1'] == category].reset_index(drop=True)
                                      for dataset in datasets]))

    # for each l1 dataset the target is now "l1_target"
    for _, datasets in l1_datasets:
        for dataset in datasets:
            dataset["target"] = dataset["L1_target"]

    # for each l2 dataset the target is now "l2_target"
    for _, datasets in l2_datasets:
        for dataset in datasets:
            dataset["target"] = dataset["L2_target"]

    # list of tuples (L1 name, dataset)
    return l1_datasets + l2_datasets


def split_data(data: pd.DataFrame, train_size: float, val_size: float, test_size: float, seed: int):
    """
    Given a dataset, split randomly into train, val, test

    :param data: Pandas Dataframe containing dataset to be split
    """
    # set seed for reproducibility
    np.random.seed(seed)

    # split along two breakpoints
    data_size = len(data)
    data_splits = np.split(data.sample(frac=1), [
        int(train_size * data_size), 
        int((train_size + val_size) * data_size)
    ])
    
    for data_split in data_splits:
        data_split.reset_index(drop=True, inplace=True)

    return data_splits
