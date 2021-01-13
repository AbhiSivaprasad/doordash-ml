import json
import csv
import pandas as pd

from typing import List, Dict, Optional, Tuple
from os.path import join


class TaxonomyNode:
    """
    Represents a category in the taxonomy

    Two types of nodes: single and group.
    A single node is associated to a model which will predict over its children.
        e.g. node: 'Meat', children: 'Chicken', 'Duck'. 
        Model associated with node will predict over its children
    A group node is associated with several models. The node itself will not be associated
    with a model but each of its children will be instead.
        e.g. node: 'Dress', children: 'Dress Length', 'Dress Formality'
             intuitively, we want to know two things about the dress so we need two models
        Each child will have a model associated with it e.g. model for dress length
    """
    def __init__(self, 
                 category_id: str,
                 category_name: str, 
                 vendor_id: Optional[str] = None,
                 model_id: Optional[str] = None,
                 group: bool = False,
                 parent: 'TaxonomyNode' = None,
                 children: List['TaxonomyNode'] = None):
        self._category_id = category_id
        self._category_name = category_name
        self._vendor_id = vendor_id
        self._model_id = model_id
        self._group = group
        self._parent = parent
        self._children = children if children is not None else []

    def add_child(self, 
                  category_id: str, 
                  category_name: str, 
                  vendor_id: Optional[str] = None,
                  model_id: Optional[str] = None,
                  group: bool = False) -> 'TaxonomyNode':
        """
        Add child node to taxonomy node with name category_name
        """
        if group and model_id:
            raise ValueError("If node is a group, there is no associated model")

        child = TaxonomyNode(category_id=category_id,
                             category_name=category_name,
                             vendor_id=vendor_id,
                             model_id=model_id,
                             group=group,
                             parent=self)

        # add child node and return it
        self.children.append(child)
        return child

    def remove(self) -> None:
        """Remove node from taxonomy"""
        self.parent.remove_child(self.category_id)
    
    def remove_child(self, category_id: str) -> None:
        """
        Remove child node from taxonomy with id category_id
        """
        index = self._get_child_index(category_id)

        if index is not None:
            del self.children[index]
        else:
            raise ValueError(f"Node with id {category_id} not found")

    def get_child(self, category_id: str) -> 'TaxonomyNode' or None:
        """Return child with category id or None if doesn't exist"""
        index = self._get_child_index(category_id)
        return self.children[index] if index is not None else None

    def _get_child_index(self, category_id: str) -> int or None:
        matches = [i for i, c in enumerate(self.children) 
                   if c.category_id == category_id]

        # category id is unique over children
        assert len(matches) <= 1

        return matches[0] if len(matches) == 1 else None

    def has_child(self, category_id: str) -> bool:
        """Return whether node has a child with id category_id"""
        return any(map(lambda c: c.category_id == category_id, self._children))

    @property
    def category_id(self):
        return self._category_id

    @property
    def category_name(self):
        return self._category_name

    @property
    def vendor_id(self):
        return self._vendor_id

    @property
    def model_id(self):
        return self._model_id

    @property
    def group(self):
        return self._group

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent


class Taxonomy:
    def __init__(self, root: TaxonomyNode = None):
        # create generic root node if not passed, with id 0
        self._root = (root 
                      if root is not None 
                      else TaxonomyNode(category_id=0, category_name="root"))

    def __str__(self) -> str:
        """readable represenation of taxonomy"""
        return json.dumps(self._repr(), indent=4)

    def __contains__(self, category_name: str) -> bool:
        """Returns True if category name is in taxonomy else False"""
        return self.find_node_by_name(category_name)[0] is not None

    def iter(self, skip_leaves: bool = False):
        return self._iter(self._root, [self._root], skip_leaves)

    def get_max_depth(self):
        """
        Return max depth of tree, root has depth 0
        """
        max_depth = 0
        for node, path in self.iter():
            if max_depth < len(path):
                max_depth = len(path)

        # zero-index depth 
        return max_depth - 1

    def add_path(self, 
                 path_category_ids: List[str], 
                 category_name: str, 
                 vendor_id: Optional[str] = None,
                 model_id: Optional[str] = None,
                 group: bool = False):
        """
        Add a node to the taxonomy.
        Category ids are not guaranteed to be a unique identifier. Instead, the full path of ids 
            from root to node will uniquely identify the node.

        :param path_category_ids: category id of nodes on path from root to new node
        """
        # find parent node in tree specified by path
        parent_node, _ = self.find_node_by_path(path_category_ids[:-1])

        # category id of node to add
        category_id = path_category_ids[-1]

        if parent_node is None:
            raise ValueError(f"Invalid path category ids:", path_category_ids)

        # add child
        child_node = parent_node.add_child(category_id, category_name, vendor_id, model_id, group)

        return child_node

    def remove_path(self, path_category_ids: List[str]) -> None:
        """Remove node in taxonomy with specified by a path of category ids from root to node"""
        node, _ = self.find_node_by_path(path_category_ids)
        
        if node is None:
            raise ValueError(f"No node found with category id: {category_id}")

        # node exists in tree
        node.remove() 

    def has_path(self, path_category_ids: List[str]) -> bool:
        """Check if a path of categories exist from root"""
        node, _ = self.find_node_by_path(path_category_ids)
        return node is not None

    def find_node_by_path(self, path_category_ids: List[str]) -> Tuple[TaxonomyNode, List[TaxonomyNode]]:
        """
        Category ids are not guaranteed to be a unique identifier. Instead, the full path of ids 
            from root to node will uniquely identify the node.

        return: Tuple (node, path) where path is the list of nodes from root to desired node.
                Root is not included in node path 
        """
        # first id in path must be root's
        assert path_category_ids[0] == self._root.category_id

        node = self._root  # track current node through path iteration
        path = [node]      # track current path of nodes
        for path_category_id in path_category_ids[1:]:
            node = node.get_child(path_category_id)

            # path does not exist
            if node is None:
                return None, None

            path.append(node)

        return node, path

    @classmethod
    def from_csv(self, filepath: str) -> 'Taxonomy':
        """Read csv into native data structure"""
        # read taxonomy from dir
        return Taxonomy.from_df(pd.read_csv(filepath))

    @classmethod
    def from_df(self, df: pd.DataFrame) -> 'Taxonomy':
        """Read df into native data structure"""
        # Sort by category names
        max_levels = len(df.columns)
        level_headers = [f"L{x}" for x in range(max_levels) if f"L{x}" in df.columns]

        # Ensures children will be processed after parent
        df = df.sort_values(by=level_headers, 
                            ascending=[True] * len(level_headers),
                            na_position="first")
       
        # all values in L0 columns should be the same
        root = TaxonomyNode(category_id=df["L0 ID"][0], 
                            category_name=df["L0"][0],
                            model_id=df["Model ID"][0])

        taxonomy = Taxonomy(root)

        for _, row in df.iterrows():
            # if row has L0, L1 headers, returns [0, 1]
            row_levels = [x for x in range(max_levels) 
                          if f"L{x} ID" in row and not pd.isnull(row[f"L{x} ID"])]

            # If row has L0, L1, headers, returns 1
            depth = row_levels[-1] 

            # skip row with depth 0 as it specifies root, which has already been parsed.
            if depth == 0:
                continue

            # extract node information 
            path_category_ids = [row[f"L{level} ID"] for level in row_levels]
            category_name = row[f"L{depth}"]
            model_id = row["Model ID"]
            vendor_id = int(row["Vendor ID"])
            group = row["Type"].lower() == "group"

            # add node to taxonomy
            taxonomy.add_path(path_category_ids=path_category_ids, 
                              category_name=category_name, 
                              vendor_id=vendor_id, 
                              model_id=model_id, 
                              group=group)

        # parse into native data structure
        return taxonomy


    def to_csv(self, filepath: str):
        """Write a human readable representation of taxonomy to a file in specified dir"""
        # write to a specific file name in dir
        # get max level (e.g. if L0, L1 categories returns 1)
        max_level = self.get_max_depth()

        # category names, category ids, vendor ids
        level_headers = [f"L{x}" for x in range(max_level + 1)]
        category_level_headers = [f"L{x} ID" for x in range(max_level + 1)]
        header = level_headers + category_level_headers + ["Vendor ID", "Model ID", "Type"]

        df = pd.DataFrame(columns=header)
        
        # write each path from root to node
        for category_names, category_ids, vendor_id, model_id, group in self._repr():
            row = {}
            for i, (category_name, category_id) in enumerate(zip(category_names, category_ids)):
                row[f"L{i}"] = category_name
                row[f"L{i} ID"] = category_id

            row["Vendor ID"] = vendor_id
            row["Model ID"] = model_id
            row["Type"] = "Group Category" if group else "Category"

            df = df.append(row, ignore_index=True)

        # Sort by category names
        df = df.sort_values(by=level_headers, 
                            ascending=[True] * len(level_headers),
                            na_position="first")

        # write df
        df.to_csv(filepath, index=False)

    def _iter(self, node: TaxonomyNode, path: List[TaxonomyNode], skip_leaves: bool):
        """Iterate recursively and output all nodes in tree"""
        if not skip_leaves or len(node.children) != 0:
            yield node, path

        for child in node.children:
            path.append(child)
            yield from self._iter(child, path, skip_leaves)
            path.pop()

    def _repr(self) -> List[Tuple[List[str], List[str], str, str, bool]]:
        """
        Return a representation of the taxonomy with one entry per node.
        Each entry is a list of names from root to node (excluding root), 
            a list of ids from root to node, vendor id, model id, and if node is group.
        """
        rows = []
        for node, path in self.iter():
            # L1, ..., Lx category names and ids
            names = [n.category_name for n in path]
            ids = [n.category_id for n in path]

            # add Lx node information
            rows.append((names, ids, node.vendor_id, node.model_id, node.group))

        return rows
