import wandb
import tempfile
import pandas as pd

from tap import Tap
from typing import Any, Dict
from random import randint

from src.data.taxonomy import Taxonomy, TaxonomyNode
from src.utils import set_seed


class ProcessTaxonomyArgs(Tap):
    seed: int = 0
    """Seed for reproducibility"""
    raw_path: str
    """Path to raw taxonomy csv"""
    processed_path: str
    """Path to write processed taxonomy csv"""


def process_taxonomy(args: ProcessTaxonomyArgs):
    set_seed(args.seed)
    taxonomy_data = pd.read_csv(args.raw_path)

    # For each tuple (A, B), 
    # A is an existing column which uniquely identifies a category (vendor id or name if unique)
    # B is the name of the new column of generated IDs
    column_mapping = [('L1', 'L1 Category ID'), ('L2', 'L2 Category ID')]

    # Eventually ids will be generated from db
    for unique_col, new_col in column_mapping:
        id_lookup = {}
        taxonomy_data[new_col] = taxonomy_data[unique_col].apply(
            lambda x: encode_id(x, id_lookup)
        )

    # Add an empty row for the root node. Vals will be filled in.
    taxonomy_data = pd.concat([pd.DataFrame([{}]), taxonomy_data]).reset_index(drop=True)

    # Setup L0 columns
    kwargs = {'L0': 'Grocery', 'L0 Category ID': snake_case('Grocery')}
    taxonomy_data = taxonomy_data.assign(**kwargs)

    # Model id just defaults to category id
    taxonomy_data["Model ID"] = ""
    taxonomy_data["Type"] = ""

    for i, row in taxonomy_data.iterrows():
        # Figure out what level row is e.g. if no L2 then max level is 1
        max_level = [
            x for x in range(len(row)) 
            if f"L{x}" in row and not pd.isnull(row[f"L{x}"])
        ][-1] if i > 0 else 0

        taxonomy_data.loc[i, "Model ID"] = row[f"L{max_level} Category ID"]
        taxonomy_data.loc[i, "Type"] = "Category"

    # creating taxonomy will automatically assign class ids if they don't exist
    taxonomy = Taxonomy().from_df(taxonomy_data)
    taxonomy.to_csv(args.processed_path)


def encode_id(value: Any, id_lookup: Dict[Any, int]) -> int:
    """Lookup id of value. If it doesn't exist, create a new one"""
    if pd.isnull(value):
        return value

    if value not in id_lookup:
        id_lookup[value] = snake_case(value) 

    return id_lookup[value]


def snake_case(name: str) -> str:
    """
    Generate an identifier based on name
    e.g. "Personal Care" --> "personal_care"
    """
    return "_".join(name.lower().split(" "))


if __name__ == '__main__':
    process_taxonomy(args=ProcessTaxonomyArgs().parse_args())
