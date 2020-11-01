import json

from typing import List, Dict
from os.path import join

from ..constants import TAXONOMY_FILE_NAME


class Taxonomy:
    def __init__(self, root: TaxonomyNode):
        """Create with read method"""
        self._root = root

        # category name --> L1, ..., Lx class id 
        self._category_name_to_class_ids = {} 
        self._build_category_name_to_class_ids(self._root, [])

    def __str__(self):
        # readable represenation
        return json.dumps(self._readable_repr(self._root), indent=4)
     
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
        node = self._root  # root

        # use class id to pick correct child at each level
        for class_id in class_ids:
            try:
                node = node.children[class_id]  # move to specified child category
            except KeyError:
                raise ValueError(f"Taxonomy does not have L{len(class_ids)} category with class ids:", class_ids)

        return node.category_name

    def write(self, dir_path: str):
        """Write a human readable representation of taxonomy to a file in specified dir"""
        # write to a specific file name in dir
        with open(join(dir_path, TAXONOMY_FILE_NAME), 'w') as f:
            f.write(str(self))

    @classmethod
    def read(self, dir_path: str) -> Taxonomy:
        # read taxonomy from dir
        with open(join(dir_path, TAXONOMY_FILE_NAME), 'r') as f:
            readable_taxonomy = json.load(f)

        # parse into native data structure
        return Taxonomy(taxonomy=self._read_from_readable_repr(readable_taxonomy))
 
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
   
    def _read_from_readable_repr(self, readable_repr, level=1):
        """Convert readable representation to native data structure"""
        # child representations stored in key "L3" in level 3
        children_repr = readable_repr[f"L{level}"]

        # collect node information then recurse on children
        taxonomy = TaxonomyNode(
            category_name=readable_repr["category"]
            class_id=readable_repr["class_id"]
            children=[self._read_from_readable_repr(child_repr, level + 1) 
                      for child_repr in children_repr]
        }

        return taxonomy

     def _readable_repr(self, node: TaxonomyNode, level=1):
        """Generate a readable, structured representation of the taxonomy"""
        # get representation of each child
        children_repr = [self._readable_repr(child, level + 1) for child in node.children]
        
        return {
            "class_id": node.class_id,
            "category": node.category_name,
            f"L{level}": children_repr
        }
 
    @property
    def category_id_to_name(self):
        return self._category_id_to_name

    @property
    def num_classes(self):
        return len(self.category_to_class_id)


class TaxonomyNode:
    """Represents a category in the taxonomy"""
    def __init__(self, category_name: str, class_id: int, children: List[TaxonomyNode] = None):
        self.category_name = category_name
        self.class_id = class_id
        self.children = children if children is not None else []
