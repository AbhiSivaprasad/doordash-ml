from os import makedirs
from os.path import join
from src.data.taxonomy import Taxonomy
from tempfile import TemporaryDirectory

import sys
import wandb
import numpy as np
import pandas as pd

def preprocess(write_dir: str, data_dir: str):
    # download raw data
    run = wandb.init(project="main", job_type="preprocessing")
    run.use_artifact("taxonomy-doordash:latest").download(data_dir)
    artifact = run.use_artifact("dataset-instacart-raw:latest")
    artifact.download(data_dir)

    # read dataset and remapping
    remapping = pd.read_csv(join(data_dir, "category_remapping.csv"))
    df = pd.read_csv(join(data_dir, "data.csv"))
    df = df.rename(columns={"Store Name": "Business"})

    # read taxonomy
    wandb_api = wandb.Api({"project": "main"})
    wandb_api.artifact("taxonomy-doordash:latest").download(data_dir)
    taxonomy = Taxonomy().from_csv(join(data_dir, "taxonomy.csv"))

    # remove nans
    mask = df[["Title", "Name", "URL", "Business", "Section"]].isnull().any(axis=1)
    df = df[~mask]

    # clean relevant columns
    df["Section"] = df["Section"].apply(clean_string)
    df["Title"] = df["Title"].apply(clean_string)
    df["Name"] = df["Name"].apply(clean_string)

    # de-duplicate by name
    df = df[~df.duplicated(subset="Name")]

    # clean title and de-duplicate by title
    df["Title"] = df["Title"].apply(lambda title: "from".join(title.split("from")[:-1]))
    df = df[~df.duplicated(subset="Title")]

    # category remapping
    del remapping['counts']  # incorrect counts
    remapping.columns = ["index", "category", "L1", "L2"]

    # nasty, fix
    low_confidence_index = [57, 67, 136, 149, 150, 154, 167, 171, 172, 174, 179, 190, 206, 218, 221, 
                            240, 249, 274, 314, 311, 324, 336, 363, 352, 373, 368, 383, 413, 392, 409, 
                            393, 405, 452, 434, 444, 440, 451, 429, 438, 467, 486, 505, 615, 589, 640, 579, 
                            594, 584, 890, 741, 771, 824, 905, 780, 834, 727, 748, 893, 980, 986, 989]

    # filter out low confidence
    remapping = remapping.loc[~remapping["index"].isin(low_confidence_index), :]

    # clean values
    remapping.loc[:, "L1"] = remapping["L1"].apply(clean_string)
    remapping.loc[:, "L2"] = remapping["L2"].apply(clean_string)
    remapping.loc[:, "category"] = remapping["category"].apply(clean_string)
    
    # remove multimatch and other labels
    remapping = remapping[~(remapping[["L1", "L2"]].isin(["MULTIMATCH", "OTHER"])).any(1)]
    
    # key = google category, value = native (L1, L2) categories
    category_mapping = {}
    for i, row in remapping.iterrows():
        category_mapping[row["category"]] = (row["L1"], row["L2"])
    
    # validate mapping
    passed_val = True
    for l1, l2 in category_mapping.values():
        # warning will fail if two nodes have same name
        l1_node = [node for node, _ in taxonomy.iter() if node.category_name == l1]
        l2_node = [node for node, _ in taxonomy.iter() if node.category_name == l2]

        assert len(l1_node) == 1
        assert len(l2_node) == 1

        l1_node = l1_node[0]
        l2_node = l2_node[0]

        if not taxonomy.has_path(['grocery', l1_node.category_id, l2_node.category_id]):
            print(f"BAD PATH: ({l1_node.category_id}, {l2_node.category_id})")
            passed_val = False

    if not passed_val:
        sys.exit(1)

    # add native categorization to data
    df["L1"] = df["Section"].apply(lambda x: get_native_category_from_instacart_section(category_mapping, x)[0])
    df["L2"] = df["Section"].apply(lambda x: get_native_category_from_instacart_section(category_mapping, x)[1])

    # drop data without categories
    df = df[df['L1'].notna() & df['L2'].notna()]

    # get L1, L2 ID directly from name (TODO: HACK!!)
    df["L1 ID"] = df["L1"].apply(get_category_id)
    df["L2 ID"] = df["L2"].apply(get_category_id)

    # select columns
    df = df[["Business", "Title", "L1", "L2", "L1 ID", "L2 ID"]]
    df.columns = ["Business", "Name", "L1", "L2", "L1 ID", "L2 ID"]

    # split test/train
    data_splits = np.split(df.sample(frac=1), [
        int(0.9 * len(df))
    ])

    for data_split in data_splits:
        data_split.reset_index(drop=True, inplace=True)

    df_train, df_test = data_splits

    # resets row numbers
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # split dataset per category
    train_datasets = split_dataset(df_train, taxonomy)
    test_datasets = split_dataset(df_test, taxonomy)
    assert len(train_datasets) == len(test_datasets)

    # create dirs for train/test
    write_data_split(train_datasets, write_dir, "train")
    write_data_split(test_datasets, write_dir, "test")

    # write whole dataset directly as files
    df_train.to_csv(join(write_dir, "all_train.csv"), index=False)
    df_test.to_csv(join(write_dir, "all_test.csv"), index=False)

    # upload artifact
    artifact = wandb.Artifact("dataset-instacart-processed", type="source-dataset")
    artifact.add_dir(write_dir)
    run.log_artifact(artifact)

def write_data_split(data_split, write_dir: str, split_name: str):
    data_split_dir = join(write_dir, split_name)
    makedirs(data_split_dir)

    for category_id, dataset in data_split.items():
        category_dir = join(data_split_dir, category_id)
        makedirs(category_dir)

        # write dataset
        dataset.to_csv(join(category_dir, f"{split_name}.csv"), index=False)

 
def split_dataset(df, taxonomy):
    datasets = {}  # key = category id, value = dataset
    for node, path in taxonomy.iter(skip_leaves=True):
        # first node in path (root) has depth 0
        depth = len(path) - 1 

        # If depth = 0 (root) then we want all L1s which is the entire dataset
        dataset = df[df[f"L{depth} ID"] == node.category_id] if depth > 0 else df
        dataset = dataset[["Business", f"L{depth + 1}", f"L{depth + 1} ID", "Name"]]
        dataset.columns = ["Business", "Category Name", "Category ID", "Name"]
        datasets[node.category_id] = dataset
    
    return datasets


def clean_string(string):
    string = " ".join(string.split(" "))
    string = string.rstrip().lstrip()
    return string

def get_category_id(name: str):
    with_underscores = "_".join(name.lower().split(" "))
    with_legal_chars = with_underscores.replace("&", "and")
    return with_legal_chars

def get_native_category_from_instacart_section(category_mapping, section):
    last_parts = section.split(">")[-2:]
    key = ">".join(last_parts).lstrip().rstrip()
    result = category_mapping.get(key, (np.nan, np.nan))
    return result

if __name__ == "__main__":
    temp_data_dir = TemporaryDirectory()
    data_dir = temp_data_dir.name

    temp_write_dir = TemporaryDirectory()
    write_dir = temp_write_dir.name

    preprocess(write_dir, data_dir)

