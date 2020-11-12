import pandas as pd
import numpy as np

from typing import List

from ..args import TrainArgs


def generate_datasets(train_data: pd.DataFrame, 
                      valid_data: pd.DataFrame, 
                      test_data: pd.DataFrame,
                      categories: List[str]):
    """
    Given a dataset, generate subsets for each category. Currently hardcode L1, L2.
    """
    datasets = [train_data, valid_data, test_data]

    # "L2" is shorthand for all L2 categories 
    if "L2" in categories:
        categories.remove("L2")
        categories.extend(list(set(train_data["L1"])))

    # extract L1 datasets
    l1_datasets = (
        # "root" is the category name for L1 in the taxonomy
        [("root", [dataset for dataset in datasets])]
        if "L1" in categories else []
    )

    # extract L2 datasets
    l2_datasets = []
    for category in [c for c in categories if c != "L1"]:
        l2_datasets.append(
            (category, [dataset[dataset['L1'] == category].reset_index(drop=True) 
                        for dataset in datasets])
        )

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


def split_data(data: pd.DataFrame, args: TrainArgs) -> List[pd.DataFrame]:
    """
    Given a dataset, split randomly into train, val, test

    :param data: Pandas Dataframe containing dataset to be split
    """
    # set seed for reproducibility
    np.random.seed(args.seed)

    # split along two breakpoints obtaining train, val, test
    data_size = len(data)
    data_splits = np.split(data.sample(frac=1), [
        int(args.train_size * data_size), 
        int((args.train_size + args.valid_size) * data_size)
    ])
    
    for data_split in data_splits:
        data_split.reset_index(drop=True, inplace=True)

    return data_splits
