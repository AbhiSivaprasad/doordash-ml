import validators
import pandas as pd
from src.preprocess.utils import hash_string
from src.preprocess.image_utils import download_images_from_urls
import os
from pathlib import Path
import numpy as np
from tap import Tap


class PredictionProcessArgs(Tap):
    raw_dir: str
    """path to file"""
    processed_dir: str
    """write processed files to dir"""
    image_dir: str = "local/images/new"
    """Dir to download images to"""
    write_path: str = "processed.csv"
    """Output file path"""
    item_name_key: str = "item_name"
    """Name of CSV key with item name"""
    image_url_key = None
    """Name of CSV key with image url"""


def hashname(filename: str):
    parts = filename.split(".")
    extension = parts[-1]
    stripped_filename = ".".join(parts[:-1])

    if stripped_filename == "":
        raise Exception("bad")

    # if the stripped name is "" then extension not found
    return f"{hash_string(stripped_filename)}.{extension}"


def preprocess(args: PredictionProcessArgs):
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    for raw_name in os.listdir(args.raw_dir):
        raw_path = os.path.join(args.raw_dir, raw_name)
        df = pd.read_csv(raw_path)
        df["Name"] = df[args.item_name_key]
        df.to_csv(os.path.join(args.processed_dir, raw_name), index=False)

    # df = pd.read_csv(args.filepath, encoding="Windows-1252")
    # df = pd.read_csv(args.filepath)

    # remap key names
    #renaming = {}

    # rename important columns
    #if args.image_url_key is not None:
    #    renaming[args.image_url_key] = "Image URL" 

    #df = df.rename(columns=renaming)

   #  if "Image URL" in df:
   #      # compute image file names as a hash of urls
   #      df["Image Name"] = df["Image URL"].apply(
   #          lambda x: hashname(x) if not pd.isnull(x) else np.nan)

   #      # validate image URLs
   #      valid_urls_mask = df["Image URL"].apply(lambda x: bool(validators.url(str(x))))

   #      # make image dir if it doesn't exist
   #      Path(args.image_dir).mkdir(exist_ok=True, parents=True)

   #      # skip already downloaded images if any
   #      current_images = [f for root, dirs, files in os.walk(image_dir) for f in files]
   #      downloaded_mask = df["Image Name"].isin(set(current_images))

   #      # download images
   #      print("Downloading images...")
   #      bad_urls = download_images_from_urls(list(df.loc[valid_urls_mask & ~downloaded_mask]["Image URL"]),
   #                                           image_dir,
   #                                           list(df.loc[valid_urls_mask & ~downloaded_mask]["Image Name"]))

   #      # remove bad df image urls
   #      bad_urls_mask = df["Image URL"].isin(set(bad_urls))
   #      df.loc[bad_urls_mask, "Image URL"] = np.nan
   #      df.loc[bad_urls_mask, "Image Name"] = np.nan



if __name__ == '__main__':
    preprocess(args=PredictionProcessArgs().parse_args())
