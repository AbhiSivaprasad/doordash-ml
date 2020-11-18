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
        Each child will have a model associated with it e.g. model for dress length
    """
    def __init__(self, 
                 category_id: str,
                 category_name: str, 
                 class_id: Optional[int] = None, 
                 vendor_id: Optional[str] = None,
                 model_id: Optional[str] = None,
                 group: bool = False,
                 parent: 'TaxonomyNode' = None,
                 children: List['TaxonomyNode'] = None):
        """
        :param class_id: Class id for category, None for root
        """
        self._category_id = category_id
        self._category_name = category_name
        self._class_id = class_id
        self._vendor_id = vendor_id
        self._model_id = model_id
        self._group = group
        self._parent = parent
        self._children = children if children is not None else []

    def add_child(self, 
                  category_id: str, 
                  category_name: str, 
                  class_id: Optional[int] = None, 
                  vendor_id: Optional[str] = None,
                  model_id: Optional[str] = None,
                  group: bool = False):
        """
        Add child node to taxonomy node with name category_name

        Class id is auto-assigned as new index in self.children if not passed
        """
        if group and model_id:
            raise ValueError("If node is a group, there is no associated model")

        child = TaxonomyNode(category_id=category_id,
                             category_name=category_name,
                             class_id=class_id if class_id is not None else len(self.children),
                             vendor_id=vendor_id,
                             model_id=model_id,
                             group=group,
                             parent=self)

        # add child node and return it
        self.children.append(child)
        return child
    
    def remove_child(self, category_id: str):
        """
        Remove child node from taxonomy with id category_id

        Class ids of children will be shifted to stay consecutive
        """
        matches = [i for i, _ in enumerate(self._children) 
                   if c._category_id == category_id]

        if len(matches) == 0:
            raise ValueError(f"Node with id {category_id} not found")

        # id must be unique
        assert len(matches) == 1

        # remove child
        index = matches[0]
        del self.children[index]

        # if node is not a a group, it has class indices
        # shift class indices to keep them consistent
        if not self.group:
            for i in range(index, len(self.children)):
                self.children[i].class_id -= 1

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
    def class_id(self):
        return self._class_id

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

    def get_max_level(self):
        """
        Return max level (L1, L2, etc.)
        Equivalent to the max depth - 1 (root)
        """
        max_level = 0
        for node, path in self.iter():
            if max_level < len(path):
                max_level = len(path)

        # don't count root in depth
        return max_level - 1

    def add(self, 
            parent_category_id: str, 
            category_id: str, 
            category_name: str, 
            class_id: Optional[int] = None, 
            vendor_id: Optional[str] = None,
            model_id: Optional[str] = None,
            group: bool = False):
        """
        Add child node to taxonomy, child's class id is assigned as index in parent's children
        """
        parent_node, _ = self.find_node_by_id(parent_category_id)

        if parent_node is None:
            raise ValueError(f"Parent node not found with category id: {parent_category_id}")

        # create child node and add to parent
        child_node = parent_node.add_child(
            category_id, category_name, class_id, vendor_id, model_id, group)

        return child_node

    def remove(self, category_id: str) -> None:
        """Remove node in taxonomy with id category_id"""
        node, path = self.find_node_by_id(category_id)
        
        if node is None:
            raise ValueError(f"No node found with category id: {category_id}")

        # if the node path has length one, the parent must be the root
        parent_node = path[-2]
        parent_node.remove_child(node.category_id)
        
    def has_link(self, parent_category_id: str, child_category_id: str) -> bool:
        """Find nodes in self which are not in other"""
        parent_node, _ = self.find_node_by_id(parent_category_id)

        # could not find parent node
        if parent_node is None:
            return False

        # check if parent has specified child
        return parent_node.has_child(child_category_id)

    def find_node_by_id(self, category_id: str) -> Tuple[TaxonomyNode, List[TaxonomyNode]]:
        """
        Search through taxonomy to find node with category id category_id
        return: Tuple (node, path) where path is the list of nodes from root to desired node.
                Root is not included in node path 
        """
        for node, path in self.iter():
            if node.category_id == category_id:
                return node, path

        return None, None

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
            depth = 1
            while f"L{depth}" in row and not pd.isnull(row[f"L{depth}"]):
                depth += 1
            depth -= 1

            # row specifies root which has already been parsed
            if depth == 0:
                continue

            # extract node information 
            parent_id = row[f"L{depth - 1} ID"]
            category_id = row[f"L{depth} ID"]
            category_name = row[f"L{depth}"]
            model_id = row["Model ID"]
            class_id = int(row["Class ID"]) if "Class ID" in row else None
            vendor_id = int(row["Vendor ID"])
            group = row["Type"].lower() == "group"

            # add node to taxonomy
            taxonomy.add(parent_category_id=parent_id, 
                         category_id=category_id, 
                         category_name=category_name, 
                         class_id=class_id, 
                         vendor_id=vendor_id, 
                         model_id=model_id, 
                         group=group)

        # parse into native data structure
        return taxonomy


    def to_csv(self, filepath: str):
        """Write a human readable representation of taxonomy to a file in specified dir"""
        # write to a specific file name in dir
        # get max level (e.g. if no L3 categories return 2)
        max_level = self.get_max_level()

        # category names, category ids, vendor ids, class id
        level_headers = [f"L{x}" for x in range(max_level + 1)]
        category_level_headers = [f"L{x} ID" for x in range(max_level + 1)]
        header = level_headers + category_level_headers + ["Vendor ID", "Class ID", "Model ID", "Type"]

        df = pd.DataFrame(columns=header)
        
        # write each path from root to node
        for category_names, category_ids, vendor_id, class_id, model_id, group in self._repr():
            row = {}
            for i, (category_name, category_id) in enumerate(zip(category_names, category_ids)):
                row[f"L{i}"] = category_name
                row[f"L{i} ID"] = category_id

            row["Vendor ID"] = vendor_id
            row["Class ID"] = class_id
            row["Model ID"] = model_id
            row["Type"] = "Group Category" if group else "Category"

            df = df.append(row, ignore_index=True)

        # Sort by category names
        df = df.sort_values(by=level_headers, 
                            ascending=[True] * len(level_headers),
                            na_position="first")

        # change cols auto inferred as floats to ints
        df["Class ID"] = df["Class ID"].astype(pd.Int64Dtype())

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

    def _repr(self) -> List[Tuple[List[str], List[str], str, str, str, bool]]:
        """
        Return a representation of the taxonomy with one entry per node.
        Each entry is a list of names from root to node (excluding root), 
            a list of ids from root to node, and class id.
        """
        rows = []
        for node, path in self.iter():
            # L1, ..., Lx category names and ids
            names = [n.category_name for n in path]
            ids = [n.category_id for n in path]

            # add Lx node information
            rows.append((names, ids, node.vendor_id, node.class_id, node.model_id, node.group))

        return rows
