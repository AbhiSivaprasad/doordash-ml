import pandas as pd
from typing import Dict
from tqdm import tqdm


def preprocess(read_path: str, write_path: str):
    # TODO: fix mislabeled categories
    # read full data
    df = pd.read_csv(read_path)

    # filter relevant columns
    df = df[['item_name', 'l1', 'l2']]
    
    # rename columns
    df.columns = ['Name', 'L1', 'L2']

    # Baby & Child has been mislabeled as Baby
    df['L1'] = df['L1'].apply(lambda x: "Baby & Child" if str(x) == "Baby" else x)

    # Prepare class variable for L1 classifier
    l1_class_name_to_id = {}
    df['L1_target'] = df['L1'].apply(lambda x: encode_target(x, l1_class_name_to_id))

    # Prepare class variable for L2 classifiers
    l1_categories = list(set(df['L1']))
    for l1_category in tqdm(l1_categories):
        l2_class_name_to_id = {}
        df.loc[df['L1'] == l1_category, 'L2'].apply(lambda x: encode_target(x, l2_class_name_to_id))
    
        for l2_category, class_id in l2_class_name_to_id.items():
            df.loc[(df['L1'] == l1_category) & (df['L2'] == l2_category), 'L2_target'] = class_id
    
    # convert float class ids to ints
    df['L1_target'] = df['L1_target'].astype(int) 
    df['L2_target'] = df['L2_target'].astype(int) 

    # clean item names
    df['Name'] = df['Name'].apply(lambda x: clean_string(str(x)))

    # write processed data
    df.to_csv(write_path, index=False)


# encode target variable
def encode_target(target: str, class_name_to_id: Dict[str, int]):
    if target not in class_name_to_id.keys():
        class_name_to_id[target] = len(class_name_to_id)

    return class_name_to_id[target]


# clean data
def clean_string(string: str):
    string = " ".join(string.split())  # standardize whitespace
    string = string.rstrip().lstrip()  # strip left/right whitespace
    return string


if __name__ == "__main__":
    preprocess("../data/doordash.csv", "../data/processed/doordash.csv")
