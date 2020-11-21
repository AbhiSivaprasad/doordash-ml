import sys
import wandb
import random
import numpy as np
import pandas as pd

from tempfile import TemporaryDirectory
from typing import Dict
from tqdm import tqdm
from os import makedirs
from os.path import join
from tap import Tap

from src.api.wandb import get_latest_artifact_identifier
from src.data.taxonomy import Taxonomy
from src.utils import set_seed


class PreprocessArgs(Tap):
    write_dir: str = None
    """Path to dir to write processed dataset files"""
    download_dir: str = None
    """Path to directory to download artifacts"""
    train_size: float = 0.9
    """Size of train dataset as a proportion of total dataset"""

    # W&B args
    upload_wandb: bool = False
    """Upload processed dataset to W&B"""
    project: str = "main"
    """Name of project in W&B"""
    raw_dataset_artifact_identifier: str = "dataset-doordash-raw:latest"
    """Name of raw dataset artifact in W&B"""
    taxonomy_artifact_identifier: str = "taxonomy-doordash:latest"
    """Name of taxonomy artifact in W&B"""
    processed_dataset_artifact_name: str = "dataset-doordash-processed"
    """Name of processed dataset artifact to create"""

    def process_args(self):
        super(PreprocessArgs, self).process_args()

        # Prevents the temporary directory from being deleted upon function return
        global temp_download_dir, temp_write_dir

        # Create temporary directory as download directory if not provided
        if self.download_dir is None:
            temp_download_dir = TemporaryDirectory()
            self.download_dir = temp_download_dir.name

        # Create temporary directory as write directory if not provided
        if self.write_dir is None:
            temp_write_dir = TemporaryDirectory()
            self.write_dir = temp_write_dir.name


def preprocess(args: PreprocessArgs):
    # set seed for reproducibility
    set_seed(1)

    api = wandb.Api({"project": args.project})
    run = wandb.init(project=args.project, job_type="preprocessing")

    # download raw dataset to specified dir path
    run.use_artifact(args.raw_dataset_artifact_identifier).download(args.download_dir)
    run.use_artifact(args.taxonomy_artifact_identifier).download(args.download_dir)

    # read full data
    df = pd.read_csv(join(args.download_dir, 'data.csv'))

    # filter & rename columns
    # drop vendor ids, they're unreliable. Fine because names are unique
    df = df[['Business', 'item_name', 'l1', 'l2']]
    df.columns = ['Business', 'Name', 'L1', 'L2']

    # add data source identifier (wandb id of processed dataset)
    latest_artifact_identifier = get_latest_artifact_identifier(
        api, artifact_identifier=f"{args.processed_dataset_artifact_name}:latest")
    
    # If no artifact found then v0 will be created
    if latest_artifact_identifier is None:
        latest_artifact_identifier = f"{args.processed_dataset_artifact_name}:v0"
    else:
        parts = latest_artifact_identifier.split(":")
        version_alias = parts.pop()
        version_number = int(version_alias[1:])
        latest_artifact_identifier = ":".join(parts + [f"v{version_number + 1}"])

    # full artifact identifier with version of processed dataset, which is about to be created, assigned as source
    df = df.assign(Source=f"{args.project}/{latest_artifact_identifier}")

    # clean mislabeled categories and ensure alignment with taxonomy
    fix_mislabeled_categories(df)
 
    # build taxonomy
    taxonomy = Taxonomy.from_csv(join(args.download_dir, 'taxonomy.csv'))

    # add category ids to data with the taxonomy (match through vendor id)
    successful = add_category_ids(df, taxonomy) 
    if not successful:
        print("Fix invalid category names!")
        sys.exit(1)

    # catch invalid class labels 
    align_to_taxonomy(df, taxonomy)

    # warn categories with less than threshold examples
    warn_small_categories(df, taxonomy, threshold=5)
    
    # clean item names
    df['Name'] = df['Name'].apply(lambda x: clean_string(str(x)))

    # split by business for better train/test distribution
    test_mask = df["Business"].isin(["7-Eleven", "CVS", "Kroger"])
    df_train = df[~test_mask]
    df_test = df[test_mask] 

    # resets row numbers
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # split dataset per category
    train_datasets = split_dataset(df_train, taxonomy)
    test_datasets = split_dataset(df_test, taxonomy)
    assert len(train_datasets) == len(test_datasets)

    # create dirs for train/test
    write_data_split(train_datasets, args.write_dir, "train")
    write_data_split(test_datasets, args.write_dir, "test")

    # write whole dataset directly as files
    df_train.to_csv(join(args.write_dir, "all_train.csv"))
    df_test.to_csv(join(args.write_dir, "all_test.csv"))

    # log processed dataset to W&B
    if args.upload_wandb:
        artifact = wandb.Artifact(args.processed_dataset_artifact_name, type='source-dataset')
        artifact.add_dir(args.write_dir)
        run.log_artifact(artifact)

        # log per category artifacts
        for category_id in train_datasets.keys():
            artifact = wandb.Artifact(f"dataset-{category_id}", type='category-dataset')
            artifact.add_file(join(args.write_dir, "train", category_id, "train.csv"))
            artifact.add_file(join(args.write_dir, "test", category_id, "test.csv"))
            run.log_artifact(artifact)


def write_data_split(data_split, write_dir: str, split_name: str):
    data_split_dir = join(write_dir, split_name)
    makedirs(data_split_dir)

    for category_id, dataset in data_split.items():
        category_dir = join(data_split_dir, category_id)
        makedirs(category_dir)

        # write dataset
        dataset.to_csv(join(category_dir, f"{split_name}.csv"))
 

def fix_mislabeled_categories(df):
    # e.g. Baby & Child has been mislabeled as Baby
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


def add_category_ids(df, taxonomy) -> bool:
    """Add category ids to df and return True if successful else False if invalid categories found"""
    # build map of category name to id
    category_name_to_category_id_mapping = {}
    for node, path in taxonomy.iter():
        category_name_to_category_id_mapping[node.category_name] = node.category_id
   
    # store set of invalid categories to not reprint
    invalid_categories = set()  # values = (category name, level)

    # function to map from category name to category id
    def category_name_to_category_id(category_name: str, level_header: str) -> str:
        try:
            return category_name_to_category_id_mapping[category_name]
        except KeyError:
            # track invalid categories
            invalid_categories.add((category_name, level_header))
            return np.nan
 
    # L1, ..., Lx
    max_levels = len(df.columns)
    level_headers = [f"L{x}" for x in range(max_levels) 
                     if f"L{x}" in df.columns]
    
    # Create L1, ..., Lx category ids
    for level_header in level_headers:
        # header for column with category id for level
        category_id_header = f"{level_header} ID"
        
        # initialize category id column
        df[category_id_header] = df[level_header].apply(
            lambda category_name: category_name_to_category_id(category_name, level_header)
        )

    # print invalid categories
    for category_name, level_header in invalid_categories:
        print(f"Invalid {level_header} Category: {category_name}")

    return len(invalid_categories) == 0


def align_to_taxonomy(df, taxonomy):
    # get full list of level headers L1, ..., Lx
    max_levels = len(df.columns)
    level_headers = [f"L{x} ID" for x in range(max_levels) 
                     if f"L{x} ID" in df.columns]

    # unique L1, ..., Lx
    unique_categories = df.drop_duplicates(level_headers)

    for _, row in unique_categories.iterrows():
        # get path of ids from L1, ..., Lx
        path_ids = [row[header] for header in level_headers 
                    if not pd.isnull(row[header])]

        # root's id must be first in path
        path_ids.insert(0, taxonomy._root.category_id)
        
        if not taxonomy.has_path(path_ids):
            print("Bad Category Path:", path_ids)


def split_dataset(df, taxonomy):
    datasets = {}  # key = category id, value = dataset
    for node, path in taxonomy.iter(skip_leaves=True):
        # first node in path (root) has depth 0
        depth = len(path) - 1 

        # If depth = 0 (root) then we want all L1s which is the entire dataset
        dataset = df[df[f"L{depth} ID"] == node.category_id] if depth > 0 else df
        dataset = dataset[["Business", f"L{depth + 1}", f"L{depth + 1} ID", "Name", "Source"]]
        dataset.columns = ["Business", "Category Name", "Category ID", "Name", "Source"]
        datasets[node.category_id] = dataset
    
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
