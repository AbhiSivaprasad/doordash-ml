import tempfile
import pandas as pd

from tap import Tap
from typing import Any, Dict
from random import randint

from src.data.taxonomy import Taxonomy


class ProcessTaxonomyArgs(Tap):
    raw_path: str
    """Path to raw taxonomy csv"""
    processed_path: str
    """Path to write processed taxonomy csv"""


def process_taxonomy(args: ProcessTaxonomyArgs):
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

    # creating taxonomy will automatically assign class ids if they don't exist
    taxonomy = Taxonomy().from_df(taxonomy_data)
    taxonomy.to_csv(args.processed_path)


def encode_id(value: Any, id_lookup: Dict[Any, int]) -> int:
    """Lookup id of value. If it doesn't exist, create a new one"""
    if pd.isnull(value):
        return value
    elif value not in id_lookup:
        id_lookup[value] = generate_id(8)

    return id_lookup[value]


def generate_id(length: int) -> int:
    """Generate random integer id with length digits"""
    return int(''.join(
        [str(randint(1, 9))] + [str(randint(0, 9)) for _ in range(length - 1)]
    ))


if __name__ == '__main__':
    process_taxonomy(args=ProcessTaxonomyArgs().parse_args())
