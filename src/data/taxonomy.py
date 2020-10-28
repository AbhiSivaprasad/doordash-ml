import json

from typing import List, Dict
from os.path import join

from ..constants import TAXONOMY_FILE_NAME


# TODO: convert to class
def write_taxonomy(l2_class_ids: List[int], l2_category_names: List[str], dir_path: str):
    """Given data, extract taxonomy. Currently handles L2 categories"""
    assert(len(l2_class_ids) == len(l2_category_names)) 

    taxonomy = [{
        'class_id': class_id, 
        'category': category
    } for class_id, category in zip(l2_class_ids, l2_category_names)]

    # write taxonomy as json
    with open(join(dir_path, TAXONOMY_FILE_NAME), 'w') as f:
        json.dump(taxonomy, f, indent=4)


def read_taxonomy(dir_path) -> Dict[str, int]:
    """read taxonomy and transform for convenience"""
    # read taxonomy from dir
    with open(join(dir_path, TAXONOMY_FILE_NAME), 'w') as f:
        taxonomy = json.load(f)
    
    # transform into direct map {category name : class id}
    taxonomy['category_to_class_id'] = {entry['category']: entry['class_id'] 
                                        for entry in taxonomy['category_to_class_id']}

    return taxonomy 
