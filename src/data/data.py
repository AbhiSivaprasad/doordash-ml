import pandas as pd
import numpy as np


def generate_datasets(train_data: pd.DataFrame, 
                      valid_data: pd.DataFrame, 
                      test_data: pd.DataFrame,
                      l1_categories: List[str]):
    """
    Given a dataset, generate subsets for each category
    """
    datasets = [train_data, valid_data, test_data]

    # for now hardcode l1, l2 
    l1_datasets = [("All", dataset.copy()) for dataset in datasets]
    l2_datasets = [
        (category, [dataset[dataset['l1'] == category].copy() for dataset in datasets])
        for category in l1_categories
    ]

    # list of tuples (L1 name, dataset)
    return l1_datasets + l2_datasets


def split_data(data: pd.DataFrame, train_size: float, val_size: float, test_size: float):
    """
    Given a dataset, split randomly into train, val, test

    :param data: Pandas Dataframe containing dataset to be split
    """
    data_size = len(data)

    # split along two breakpoints
    return np.split(data.sample(frac=1), [
        int(train_size * data_size), 
        int((train_size + val_size) * data_size]
    )
