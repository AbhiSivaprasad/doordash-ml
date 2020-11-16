import wandb
import random
import numpy as np
import pandas as pd

from tempfile import TemporaryDirectory
from typing import Dict
from tqdm import tqdm
from os.path import join
from tap import Tap

from src.data.taxonomy import Taxonomy
from src.utils import set_seed


class PreprocessArgs(Tap):
    write_dir: str = "data/doordash/processed"
    """Path to dir to write processed dataset files"""
    train_size: float = 0.9
    """Size of train dataset as a proportion of total dataset"""

    # W&B args
    upload_wandb: bool = False
    """Upload processed dataset to W&B"""
    project: str = "doordash"
    """Name of project in W&B"""
    download_dir: str
    """Path to directory to download artifacts"""
    artifact_name: str = "raw-dataset:latest"
    """Name of raw dataset artifact in W&B"""
    taxonomy_artifact_name: str = "taxonomy:latest"
    """Name of taxonomy artifact in W&B"""

    def process_args(self):
        super(CommonArgs, self).process_args()

        # Create temporary directory as save directory if not provided
        global temp_dir  # Prevents the temporary directory from being deleted upon function return
        if self.download_dir is None:
            temp_dir = TemporaryDirectory()
            self.download_dir = temp_dir.name


def preprocess(args: PreprocessArgs):
    # set seed for reproducibility
    set_seed(1)

    run = wandb.init(project=args.project, job_type="preprocessing")

    # download raw dataset to specified dir path
    run.use_artifact(args.artifact_name).download(args.download_dir)
    run.use_artifact(args.taxonomy_artifact_name).download(args.download_dir)

    # read full data
    df = pd.read_csv(join(args.download_dir, 'data.csv'))

    # filter & rename columns
    df = df[['item_name', 'l1', 'l2', 'category1_tag_id', 'category2_tag_id']]
    df.columns = ['Name', 'L1', 'L2', 'L1 ID', 'L2 ID']

    # clean mislabeled categories and ensure alignment with taxonomy
    fix_mislabeled_categories(df)
 
    # build taxonomy
    taxonomy = Taxonomy.from_csv(join(args.download_dir, 'taxonomy.csv'))

    # add category ids to data with the taxonomy (match through vendor id)
    add_category_ids(df, taxonomy) 

    # catch invalid class labels 
    align_to_taxonomy(df, taxonomy)

    # warn categories with less than threshold examples
    df = warn_small_categories(df, taxonomy, threshold=5)
    
    # clean item names
    df['Name'] = df['Name'].apply(lambda x: clean_string(str(x)))

    import sys
    sys.exit(1)

    # split in train, test
    df_train, df_test = np.split(df.sample(frac=1), [
        int(args.train_size * len(df)), 
    ])


    # write processed data
    df_train.to_csv(join(args.write_dir, "train.csv"), index=False)
    df_test.to_csv(join(args.write_dir, "test.csv"), index=False)

    # log processed dataset to W&B
    if args.upload_wandb:
        artifact = wandb.Artifact('processed-dataset', type='dataset')
        artifact.add_dir(args.write_dir)
        run.log_artifact(artifact)


def fix_mislabeled_categories(df):
    # Baby & Child has been mislabeled as Baby
    mislabeled_l1_categories_as_l1 = {
        "Baby": "Baby & Child",
        "Condiments": "Pantry",
        "Beauty": "Personal Care"
    }

    df['L1'] = df['L1'].apply(lambda x: mislabeled_l1_categories_as_l1.get(x, x))

    # Condiment, Beauty are marked L1 but are L2
    mislabeled_l1_categories_as_l2 = {
        ("Produce", "Canned Specialty"): ("Pantry", "Canned Specialty"),
        ("Household", "Dog Treats & Toys"): ("Pet Care", "Dog Treats & Toys"),
        ('Household', 'Hand Soap'): ('Personal Care', 'Hand Soap'),
        ('Fresh Food', 'Health'): ('Fresh Food', 'Sandwiches'),
        ('Drinks', 'Ice'): ('Frozen', 'Ice'),
        ('Household', 'Liquor'): ('Alcohol', 'Liquor'),
        ('Vitamins', 'Liquor'): ('Alcohol', 'Liquor'),
        ('Frozen', 'Poultry'): ('Meat & Fish', 'Poultry'),
        ('Frozen', 'Seafood'): ('Meat & Fish', 'Seafood'),
        ('Frozen', 'Sides'): ('Fresh Food', 'Sides'),
        ('Snacks', 'Wings'): ('Fresh Food', 'Sides'),
        ('Personal Care', 'Sun care'): ('Personal Care', 'Sun Care')
    }

    df[['L1', 'L2']] = [mislabeled_l1_categories_as_l2.get((t.L1, t.L2), (t.L1, t.L2)) 
                        for t in df[['L1', 'L2']].itertuples()]


def add_category_ids(df, taxonomy):
    # iterate through taxonomy and build vendor id --> category id
    vendor_id_to_category_id = {}
    for node, path in taxonomy.iter():
        if node.vendor_id is not None:
            vendor_id_to_category_id[node.vendor_id] = node.category_id
    
    # L1, ..., Lx
    max_levels = len(df.columns)
    level_headers = [f"L{x}" for x in range(max_levels) 
                     if f"L{x}" in df.columns]
    
    # Create L1, ..., Lx category ids
    for level_header in level_headers:
        # Lx --> Lx Vendor ID
        category_id_header = f"{level_header} Vendor ID"

        # convert column of vendor ids to column of category ids
        taxonomy_data[category_id_header] = taxonomy_data[level_header].apply(
            lambda x: vendor_id_to_category_id[x]
        )


def align_to_taxonomy(df, taxonomy):
    # get full list of level headers L1, ..., Lx
    max_levels = len(df.columns)
    level_headers = [f"L{x}" for x in range(max_levels) 
                     if f"L{x}" in df.columns]

    # unique L1, ..., Lx
    unique_categories = df.drop_duplicates(level_headers)

    for categories in unique_categories.iterrows():
        for i in range(len(level_headers)):
            child_id = categories[f"L{i + 1} ID"]
            parent_id = (categories[f"L{i} ID"] 
                         if i > 0 
                         else taxonomy._root.category_id)

            if not taxonomy.has_link(parent_id, child_id):
                print(f"Bad L{i + 1} Category:", categories[f"L{i + 1}"])
                print(categories)


def split_dataset(df, taxonomy):
    datasets = []
    for node, path in taxonomy.iter(skip_leaves=True):
        # first node in path (root) has depth 0
        depth = len(path) - 1 

        # If depth = 0 (root) then we want all L1s which is the entire dataset
        dataset = df[df[f"L{depth} ID"] == node.category_id] if depth > 0 else df
        dataset = dataset[f"L{depth + 1}", f"L{depth + 1} ID", "Name"]
        datasets.append((node.category_id, dataset))
    
    return datasets


def warn_small_categories(df, taxonomy, threshold: int):
    for node, path in taxonomy.iter():
        # root node contains entire dataset, skip
        if len(path) == 1:
            continue

        # first node in path (root) has depth 0
        depth = len(path) - 1 

        # extract dataset
        size = len(df[df[f"L{depth} ID"] == node.category_id])
        if size <= threshold:
            print(f"L{depth} Category {node.category_name} has {size} examples")


# clean data
def clean_string(string: str):
    string = " ".join(string.split())  # standardize whitespace
    string = string.rstrip().lstrip()  # strip left/right whitespace
    string = string.lower()
    return string


if __name__ == "__main__":
    preprocess(args=PreprocessArgs().parse_args())
