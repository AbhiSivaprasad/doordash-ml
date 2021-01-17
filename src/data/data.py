import pandas as pd
import numpy as np

from typing import List
from collections import OrderedDict

from ..args import TrainArgs


def encode_target_variable(data: pd.DataFrame) -> List[str]:
    """
    Create a target column with class ids and return a mapping from class ids to category ids

    :return: List[str]. Labels in a list where element i is the category id for class id i
    """
    # create target variable
    category_ids = list(set(data["Category ID"))

    # sort categories for consistent labelling
    category_ids.sort()

    # element i is the category id for class id i
    category_id_to_label = {category_id: i for i, category_id in enumerate(category_ids)}

    # generate targets 
    data["target"] = data["Category ID"].apply(lambda x: category_id_to_label[x]) 

    return category_ids


def encode_target_variable_with_labels(data: pd.DataFrame, labels: List[str]) -> None:
    """
    Create a target column with class ids based on the mapping provided in labels
    
    :param labels: ith label corresponds to category id for class i
    """
    category_id_to_label = {category_id: class_id 
                            for class_id, category_id in enumerate(labels)}

    # apply mapping to create target column
    data["target"] = data["Category ID"].apply(
        lambda category_id: category_id_to_label[category_id] if category_id in category_id_to_label else -1
    )


def split_data(data: pd.DataFrame, args: TrainArgs) -> List[pd.DataFrame]:
    """
    Given a dataset, split randomly into train, val, test

    :param data: Pandas Dataframe containing dataset to be split
    """
    # set seed for reproducibility
    np.random.seed(args.seed)

    # split along two breakpoints obtaining train, val, test
    data_size = len(data)
    breakpoints = [
        int(args.train_size * data_size), 
        int((args.train_size + args.valid_size) * data_size)
    ]

    # if no test set then remove the last breakpoint
    if args.test_size == 0:
        breakpoints = breakpoints[:-1]

    data_splits = np.split(data.sample(frac=1), breakpoints)
    
    for data_split in data_splits:
        data_split.reset_index(drop=True, inplace=True)

    return data_splits
