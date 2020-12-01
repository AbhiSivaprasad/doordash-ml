import wandb

import pandas as pd
from os import makedirs, listdir
from os.path import join, isdir

from typing import List
from tap import Tap
from copy import copy
from tempfile import TemporaryDirectory
from pathlib import Path


class DownloadArgs(Tap):
    sources: List[str] = []
    """
    List of W&B artifact identifiers to source datasets
    Adds sources to both train sources and test sources
    """
    train_sources: List[str] = []
    """
    List of W&B artifact identifiers to source datasets
    The train portion of these datasets will be merged
    """
    test_sources: List[str] = []
    """
    List of W&B artifact identifiers to source datasets
    The train portion of these datasets will be merged
    """
    
    write_dir: str
    """Directory to store merged datasets"""
    save_dir: str = None
    """Directory to download source datasets"""
    wandb_project: str = "main"
    """Name of W&B project with source datasets"""

    def validate_sources(self):
        if not (self.sources or self.train_sources or self.test_sources):
            raise ValueError("At least one source must be specified")

    def process_args(self):
        super(DownloadArgs, self).process_args() 

        self.validate_sources()

        if self.sources:
            self.train_sources += self.sources.copy()
            self.test_sources += self.sources.copy()

        # Create temporary directory as save directory if not provided
        global temp_dir  # Prevents the temporary directory from being deleted upon function return
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name


def download(args: DownloadArgs):
    """Download dataset from specified sources and merge"""
    all_sources = set(args.train_sources + args.test_sources)
    
    # wandb api
    wandb_api = wandb.Api({"project": args.wandb_project})

    # download data into a folder named after source
    for source in all_sources:
        source_dir = join(args.save_dir, source)
        makedirs(source_dir)

        # download source dataset
        wandb_api.artifact(source).download(source_dir)
    
    # dir paths to source datasets
    train_source_dirs = [join(args.save_dir, train_source) 
                         for train_source in args.train_sources]

    test_source_dirs = [join(args.save_dir, test_source) 
                        for test_source in args.test_sources]

    # create separate train, test dirs to save merged datasets
    train_write_dir = join(args.write_dir, "train")
    test_write_dir = join(args.write_dir, "test")

    Path(train_write_dir).mkdir(parents=True, exist_ok=True)
    Path(test_write_dir).mkdir(parents=True, exist_ok=True)

    # merge datasets are write to output dir
    merge(train_source_dirs, train_write_dir, "train", "train.csv")
    merge(test_source_dirs, test_write_dir, "test", "test.csv")


def merge(dir_paths: List[str], write_dir: str, data_split: str = "train", data_filename: str = "train.csv"):
    # key = category_id, value = merged dataset for category
    datasets = {}

    for dir_path in dir_paths:
        data_split_path = join(dir_path, data_split)
        for category_id in listdir(data_split_path):
            # skip files
            category_dir = join(data_split_path, category_id)
            if not isdir(category_dir):
                continue
        
            # read data from /category_name/data_filename
            df = pd.read_csv(join(category_dir, data_filename))
            
            if category_id in datasets:
                # merge df with stored dataset
                df = pd.concat([datasets[category_id], df])

            # create or update dataset
            datasets[category_id] = df

    # write datasets to dir
    for category_id, dataset in datasets.items():
        makedirs(join(write_dir, category_id))
        dataset.to_csv(join(write_dir, category_id, data_filename))
