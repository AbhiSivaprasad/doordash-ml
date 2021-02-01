import validators
import pandas as pd
from src.preprocess.utils import hash_string
from src.preprocess.image_utils import download_images_from_urls
import os
from pathlib import Path
import numpy as np


def hashname(filename: str):
    parts = filename.split(".")
    extension = parts[-1]
    stripped_filename = ".".join(parts[:-1])

    if stripped_filename == "":
        raise Exception("bad")

    # if the stripped name is "" then extension not found
    return f"{hash_string(stripped_filename)}.{extension}"


def preprocess(filename: str, image_dir: str, write_path: str, item_name_key: str, image_url_key: str):
    df = pd.read_csv(filename, encoding="Windows-1252")

    # remap key names
    renaming = {}

    # rename important columns
    if item_name_key is not None:
        renaming[item_name_key] = "Name" 
    if image_url_key is not None:
        renaming[image_url_key] = "Image URL" 
    df = df.rename(columns=renaming)

    if "Image URL" in df:
        # compute image file names as a hash of urls
        df["Image Name"] = df["Image URL"].apply(
            lambda x: hashname(x) if not pd.isnull(x) else np.nan)

        # validate image URLs
        valid_urls_mask = df["Image URL"].apply(lambda x: bool(validators.url(str(x))))

        # make image dir if it doesn't exist
        Path(image_dir).mkdir(exist_ok=True, parents=True)

        # skip already downloaded images if any
        current_images = [f for root, dirs, files in os.walk(image_dir) for f in files]
        downloaded_mask = df["Image Name"].isin(set(current_images))

        # download images
        print("Downloading images...")
        bad_urls = download_images_from_urls(list(df.loc[valid_urls_mask & ~downloaded_mask]["Image URL"]),
                                             image_dir,
                                             list(df.loc[valid_urls_mask & ~downloaded_mask]["Image Name"]))

        # remove bad df image urls
        bad_urls_mask = df["Image URL"].isin(set(bad_urls))
        df.loc[bad_urls_mask, "Image URL"] = np.nan
        df.loc[bad_urls_mask, "Image Name"] = np.nan

    # write processed predictions
    df.to_csv(write_path, index=False)


if __name__ == "__main__":
    filename = "raw/Gristedes-Dagastino.csv"
    image_dir = "local/images/new/"
    write_path = "processed/Gristedes-Dagastino.csv"
    item_name_key = "item_name"
    image_url_key = None

    preprocess(filename, image_dir, write_path, item_name_key, image_url_key)
