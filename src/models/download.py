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
    category_ids: List[str] = None
    """List of category ids to train models for"""
    all_categories: bool = False
    """If True, run on all categories in taxonomy. taxonomy must be specified"""
    taxonomy: str = None
    """W&B identifier of taxonomy artifact if all_categories is selected"""

    write_dir: str
    """Directory to models"""
    wandb_project: str = "main"
    """Name of W&B project with runs and models"""
    download_dir: str = None
    """Name of temp directory to download artifacts"""

    """Pass variable args to filter by config"""

    def process_args(self):
        # Create temporary directory as save directory if not provided
        global temp_dir  # Prevents the temporary directory from being deleted upon function return
        if self.download_dir is None:
            temp_dir = TemporaryDirectory()
            self.download_dir = temp_dir.name


def download(args: DownloadArgs):
    """Download model per category given properties"""

    # extra_args holds all passed in args not specified in Args
    query_args = args.extra_args
    query = build_query(query_args) 

    # if all category ids specificed, then get taxonomy and iterate through categories
    category_ids = args.category_ids
    if args.all_categories:
        wandb_api.artifact(args.taxonomy_artifact_identifier).download(args.save_dir)
        taxonomy = Taxonomy.from_csv(join(args.save_dir, "taxonomy.csv"))
        category_ids = [node.category_id for node, _ in taxonomy.iter(skip_leaves=True)]

    for category_id in category_ids:
        # create dir for category
        category_dir = join(write_dir, category_id)
        Path(category_dir).mkdir(parents=True, exist_ok=True)

        # add category id to query
        query["config.category_id"] = category_id

        # filter runs
        runs = wandb_api.runs(path=args.wandb_project, 
                              filters=query_filters, 
                              order="+summary_metrics.test loss")
        print(f"Using run {runs[0].name} for category id {category_id}")

        # pull output artifact from run
        model_artifact = next(runs[0].logged_artifacts())
        model_artifact.download(category_dir)


def build_query(arg_parts: List[str]):
    """
    Build query from extra args
    e.g. args_parts = ["--train_datasets", "a", "b", "--model_name", "c"]
         returns: {"train_datasets": ["a", "b"], "model_name": "c"}
        
    :param arg_parts: list of parts of command line args 
    """
    args = {}
    curr_arg_name = None
    curr_arg_values = []

    for arg_part in arg_parts:
        if is_arg_name(arg_part):
            # current arg name is None for first arg
            if curr_arg_name is not None:
                add_arg(args, curr_arg_name, curr_arg_values)

            # assign new arg name "--model_name" --> "model_name"
            curr_arg_name = arg_part[2:]
            curr_arg_values = []
        else:
            curr_arg_values.append(arg_part)
    
    # flush out last arg
    add_arg(args, curr_arg_name, curr_arg_values)

    return args


def add_arg(args, arg_name, arg_values):
    if len(arg_values) == 1:
        arg_values = arg_values[0]

    args[arg_name] = arg_values


def is_arg_name(name: str):
    return len(name) > 2 and name[:2] == "--"
