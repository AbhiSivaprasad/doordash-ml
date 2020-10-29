import json

from typing import List, Dict
from os.path import join

from ..constants import TAXONOMY_FILE_NAME


class Taxonomy:
    def __init__(self, category_to_class_id: Dict[str, int]):
        self.category_to_class_id = category_to_class_id

    def write(self, dir_path: str):
        # readable represenation
        readable_taxonomy = [
            {"class_id": class_id, "category": category} 
            for category, class_id in self.category_to_class_id.items()
        ]

        with open(join(dir_path, TAXONOMY_FILE_NAME), 'w') as f:
            json.dump(readable_taxonomy, f, indent=4)
    
    @classmethod
    def read(self, dir_path: str):
        # read taxonomy from dir
        with open(join(dir_path, TAXONOMY_FILE_NAME), 'r') as f:
            readable_taxonomy = json.load(f)
            category_to_class_id = {
                item["category"]: item["class_id"] for item in readable_taxonomy
            }

        return Taxonomy(category_to_class_id=category_to_class_id)
   
    def class_id_to_category(self, class_id: int) -> str:
        for category, candidate_class_id in self.category_to_class_id.items():
            if class_id == candidate_class_id:
                return category

        raise ValueError("Taxonomy does not have class id", class_id)

    @property
    def num_classes(self):
        return len(self.category_to_class_id)
