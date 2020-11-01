from __future__ import annotations
from typing import List, Dict, Optional
from os.path import join
from ..constants import TAXONOMY_FILE_NAME

import json


class TaxonomyNode:
    """Represents a category in the taxonomy"""
    def __init__(self, category_name: str, class_id: Optional[int], children: List[TaxonomyNode] = None):
        """
        :param class_id: Class id for category, None for root
        """
        self.category_name = category_name
        self.class_id = class_id
        self.children = children if children is not None else []

    def add_child(self, category_name: str):
        """
        Add child node to taxonomy node with name category_name

        Class id is auto-assigned as new index in self.children
        """
        self.children.append(
            TaxonomyNode(category_name=category_name, class_id=len(self.children))
        )


#TODO add validity checks
class Taxonomy:
    def __init__(self, root: TaxonomyNode = None):
        # create generic root node if not passed
        self._root = (root 
                      if root is not None 
                      else TaxonomyNode(category_name="root", class_id=None))

    def __str__(self):
        # readable represenation
        return json.dumps(self._readable_repr(self._root), indent=4)

    def add(self, parent_category: str, child_category: str):
        """
        Add child node to taxonomy, child's class id is assigned as index in parent's children
        """
        parent_node, _ = self._find_node_by_name(self._root, parent_category, [])

        if parent_node is None:
            raise ValueError(f"No node found with category: {parent_category}")

        # create child node and add to parent
        parent_node.add_child(child_category)

    def category_name_to_class_ids(self, category_name: str) -> List[int]:
        """
        Given a category name which uniquely identifies a category, return L1...Lx class ids
        e.g. Categories x, y are L1, L2 categories for L3 category z then
             return [class_id(x), class_id(y), class_id(z)]
        """
        # walk through tree to find node by name (unique identifier)
        node, node_path = self._find_node_by_name(self._root, category_name, [])

        if node is None:
            raise ValueError(f"No node found with category: {category_name}")

        return [n.class_id for n in node_path]

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

    def read(self, dir_path: str) -> Taxonomy:
        """Read the human readable representation into native data structure"""
        # read taxonomy from dir
        with open(join(dir_path, TAXONOMY_FILE_NAME), 'r') as f:
            readable_taxonomy = json.load(f)

        # parse into native data structure
        return Taxonomy(root=self._read_from_readable_repr(readable_taxonomy))
 
    def write(self, dir_path: str):
        """Write a human readable representation of taxonomy to a file in specified dir"""
        # write to a specific file name in dir
        with open(join(dir_path, TAXONOMY_FILE_NAME), 'w') as f:
            f.write(str(self))

    def _find_node_by_name(self, 
                           node: TaxonomyNode, 
                           category_name: str, 
                           node_path: List[TaxonomyNode]) -> Tuple[TaxonomyNode, List[TaxonomyNode]]:
        """
        Search through taxonomy to find node with name category_name. 

        :param node_path: Node path from root to current point in recursion, call with []
        return: Tuple (node, node path) where node path is the list of nodes from root to desired node.
                Root is not included in node path 
        """
        # check if current node is desired category
        if node.category_name == category_name:
            return node, node_path

        # recurse through children looking for category name
        for child in node.children:
            node_path.append(child)
            if self._find_node_by_name(child, category_name, [])[0] is not None:
                return child, node_path
            node_path.pop()

        # category not found
        return None, None
  
    def _read_from_readable_repr(self, readable_repr, level=1):
        """Convert readable representation to native data structure"""
        # child representations stored in key "L3" in level 3
        # innermost nodes don't have children so key doesn't exist
        children_repr = (readable_repr[f"L{level}"] 
                         if f"L{level}" in readable_repr 
                         else [])

        # collect node information then recurse on children
        return TaxonomyNode(
            category_name=readable_repr["category"],
            class_id=readable_repr["class id"],
            children=[self._read_from_readable_repr(child_repr, level + 1) 
                      for child_repr in children_repr]
        )

    def _readable_repr(self, node: TaxonomyNode, level=1):
        """Generate a readable, structured representation of the taxonomy"""
        node_repr = {
            "class id": node.class_id,
            "category": node.category_name,
        }
        
        # get representation of each child
        children_repr = [self._readable_repr(child, level + 1) for child in node.children]

        if children_repr:
            node_repr[f"L{level}"] = children_repr

        return node_repr
