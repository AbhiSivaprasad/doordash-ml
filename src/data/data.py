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
    # encode category ids
    category_id_to_class_id_mapping = OrderedDict()
    def category_id_to_class_id(category_id: str) -> int:
        if category_id not in category_id_to_class_id_mapping:
            category_id_to_class_id_mapping[category_id] \
                = len(category_id_to_class_id_mapping)

        return category_id_to_class_id_mapping[category_id]

    # create target variable
    data["target"] = data["Category ID"].apply(category_id_to_class_id) 

    # class ids were added in order to mapping dict
    labels = [category_id for category_id, _ in category_id_to_class_id_mapping.items()]
    return labels


def encode_target_variable_with_labels(data: pd.DataFrame, labels: List[str]) -> None:
    """
    Create a target column with class ids based on the mapping provided in labels
    
    :param labels: ith label corresponds to category id for class i
    """
    category_id_to_label = {category_id: class_id 
                            for class_id, category_id in enumerate(labels)}

    # apply mapping to create target column
    data["target"] = data["Category ID"].apply(
        lambda category_id: category_id_to_label[category_id]
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
    data_splits = np.split(data.sample(frac=1), [
        int(args.train_size * data_size), 
        int((args.train_size + args.valid_size) * data_size)
    ])
    
    for data_split in data_splits:
        data_split.reset_index(drop=True, inplace=True)

    return data_splits
