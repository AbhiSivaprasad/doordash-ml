import wandb
import random
import numpy as np
import pandas as pd

from typing import Dict
from tqdm import tqdm
from os.path import join
from tap import Tap

from src.data.taxonomy import Taxonomy


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
    artifact_download_dir: str = "data/doordash/raw"
    """Path to directory to download raw dataset artifact"""
    artifact_name: str = "raw-dataset:latest"
    """Name of raw dataset artifact in W&B"""


def preprocess(args: PreprocessArgs):
    run = wandb.init(project=args.project, job_type="preprocessing")

    # download raw dataset to specified dir path
    run.use_artifact(args.artifact_name).download(args.artifact_download_dir)

    # set seed for reproducibility
    np.random.seed(1)
    random.seed(1)

    # TODO: do duplicate check on names
    # read full data
    df = pd.read_csv(join(args.artifact_download_dir, 'data.csv'))

    # filter & rename columns
    df = df[['item_name', 'l1', 'l2']]
    df.columns = ['Name', 'L1', 'L2']

    # clean mislabeled categories and ensure alignment with taxonomy
    fix_mislabeled_categories(df)
 
    # build taxonomy
    taxonomy = build_taxonomy(
        pd.read_csv(join(args.artifact_download_dir, 'taxonomy.csv'))
    )
    taxonomy.validate()

    # catch invalid class labels 
    align_to_taxonomy(df, taxonomy)

    # remove class variables with <= 5 examples from taxonomy and data
    df = remove_small_categories(df, taxonomy, threshold=5)
    
    # Prepare class variable for L1 classifier
    df['L1_target'] = df['L1'].apply(lambda x: taxonomy.category_name_to_class_ids(x)[0])
    df['L2_target'] = df['L2'].apply(lambda x: taxonomy.category_name_to_class_ids(x)[1])

    # convert float class ids to ints
    df['L1_target'] = df['L1_target'].astype(int) 
    df['L2_target'] = df['L2_target'].astype(int) 

    # clean item names
    df['Name'] = df['Name'].apply(lambda x: clean_string(str(x)))

    # split in train, test
    df_train, df_test = np.split(df.sample(frac=1), [
        int(args.train_size * len(df)), 
    ])

    # write processed data
    df_train.to_csv(join(args.write_dir, "train.csv"), index=False)
    df_test.to_csv(join(args.write_dir, "test.csv"), index=False)

    # write taxonomy
    taxonomy.write(join(args.write_dir))

    # log processed dataset to W&B
    if args.wandb:
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

    # print([t for t in df[['L1', 'L2']].itertuples()])
    df[['L1', 'L2']] = [mislabeled_l1_categories_as_l2.get((t.L1, t.L2), (t.L1, t.L2)) 
                        for t in df[['L1', 'L2']].itertuples()]

def align_to_taxonomy(df, taxonomy):
    #TODO: move to tree diff
    unique_categories = df[['L1', 'L2']].drop_duplicates(['L1', 'L2'])

    # check L1 categories
    for l1 in list(set(unique_categories["L1"])):
        if not taxonomy.has_link("root", l1):
            print("bad L1 category:", l1)
            print(f"\thas L2 categories:", set(unique_categories[unique_categories['L1'] == l1]['L2']))

    # add L2 categories 
    for row in unique_categories[["L1", "L2"]].itertuples():
        if not taxonomy.has_link(row.L1, row.L2):
            print("bad L2 Category. (L1, L2):", (row.L1, row.L2))


def remove_small_categories(df, taxonomy, threshold: int):
    for node in taxonomy.iter_level(2):
        category = node.category_name
        size = len(df[df["L2"] == category])
        if size <= threshold:
            print(f"Removing L2 category {category} with {size} examples")
            taxonomy.remove(category)

            # l2 category should be unique
            df = df[(df['L2'] != category)]

    return df


def build_taxonomy(df):
    # TODO: sort alphabetically
    taxonomy = Taxonomy()
    unique_categories = df[['L1', 'L2']].drop_duplicates(['L1', 'L2']).sort_values(['L1', 'L2'])
    
    # track seen l1 categories
    seen_l1 = set()

    # add L1, L2 categories 
    for row in unique_categories[["L1", "L2"]].itertuples():
        if not row.L1 in seen_l1:
            taxonomy.add("root", row.L1)
            seen_l1.add(row.L1)
        taxonomy.add(row.L1, row.L2)

    return taxonomy


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
    preprocess(args=PreprocessArgs().parse_args())
