import json

from typing import List, Dict
from os.path import join

from ..constants import TAXONOMY_FILE_NAME


class Taxonomy:
    def __init__(self, taxonomy: TaxonomyNode):
        """Create with read method"""
        self._taxonomy = taxonomy

        # category name --> L1, ..., Lx class id 
        self._category_name_to_class_ids = {} 
        self._build_category_name_to_class_ids()

   def category_name_to_class_ids(self, category_name: str) -> List[int]:
        """
        Given a category name which uniquely identifies a category, return L1...Lx class ids
        e.g. Categories x, y are L1, L2 categories for L3 category z then
             return [class_id(x), class_id(y), class_id(z)]
        """
        return self._category_name_to_class_ids[category_name]

    def class_ids_to_category_name(self, class_ids: List[int]) -> str:
        """
        A list of class ids will uniquely identify a category.
        e.g. [3]    --> L1 category with class id 3, 
             [2, 3] --> L2 category with class id 3, for L1 category with class id 2
        """
        node = self._taxonomy
        for class_id in class_ids:
            try:
                node = node.children[class_id]
            except KeyError:
                raise ValueError(f"Taxonomy does not have L{len(class_ids)} category with class ids:", class_ids)

        return node.category_name

    def write(self, dir_path: str):
        # readable represenation
        readable_taxonomy = [
            {"class_id": class_id, "category": category} 
            for category, class_id in self.category_to_class_id.items()
        ]

        with open(join(dir_path, TAXONOMY_FILE_NAME), 'w') as f:
            json.dump(readable_taxonomy, f, indent=4)
    
    @classmethod
    def read(self, dir_path: str) -> Taxonomy:
        # read taxonomy from dir
        with open(join(dir_path, TAXONOMY_FILE_NAME), 'r') as f:
            readable_taxonomy = json.load(f)
            category_to_class_id = {
                item["category"]: item["class_id"] for item in readable_taxonomy
            }

        return Taxonomy(category_to_class_id=category_to_class_id)
    
    def _build_category_name_to_class_ids(self, node: TaxonomyNode, class_ids: List[int]):
        """
        Build a map from a category name (unique identifier) and its L1,...,Lx class ids
        e.g. Categories x, y are L1, L2 categories for L3 category z then
             add entry (z, [class_id(x), class_id(y), class_id(z)])
        
        :param class_ids: class ids to assign to current category, maintained during recursion
        """
        # register current category
        if class_ids:  # skip if root
            self._category_name_to_class_ids[node.category_name] = class_ids.copy()

        # when recursing into child add child's id to current path
        for class_id, child in enumerate(node.children):
            class_ids.append(class_id)
            self._build_category_name_to_class_ids(child, class_ids)  # build subtree rooted at child
            class_ids.pop()
 
    @property
    def category_id_to_name(self):
        return self._category_id_to_name

    @property
    def num_classes(self):
        return len(self.category_to_class_id)


class TaxonomyNode:
    """Represents a category in the taxonomy"""
    def __init__(self, category_name: str, children: List[TaxonomyNode] = None):
        self.category_name = category_name
        self.children = children if children is not None else []
