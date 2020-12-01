import wandb
import pandas as pd

def prepare_raw_data():
    run = wandb.init(project="main", job_type="preprocessing")

    df = pd.read_csv("google.csv")
    remapping = pd.read_csv("google/category_remapping.csv")

    # desired categories
    category_set = set(remapping["category"])
    
    # filter dataset for categories
    pruned_df = df[df["category"].isin(category_set)]

    # write 
    pruned_df.to_csv("google/data.csv")

    artifact = wandb.Artifact("dataset-google-raw", type="source-dataset")
    artifact.add_dir("google/")
    run.log_artifact(artifact)


if __name__ == '__main__':
    prepare_raw_data()
