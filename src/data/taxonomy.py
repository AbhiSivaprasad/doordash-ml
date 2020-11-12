from typing import List, Dict, Optional, Tuple
from os.path import join
from ..constants import TAXONOMY_FILE_NAME

import json


class TaxonomyNode:
    """Represents a category in the taxonomy"""
    def __init__(self, 
                 category_name: str, 
                 class_id: Optional[int], 
                 children: List['TaxonomyNode'] = None):
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
        child = TaxonomyNode(category_name=category_name, class_id=len(self.children))
        self.children.append(child)
        return child
    
    def remove_child(self, category_name: str):
        """
        Remove child node from taxonomy with name category_name

        Class ids of children will be shifted to stay consecutive
        """
        index = -1
        for i, child in enumerate(self.children):
            if child.category_name == category_name:
                index = i

        if index != -1:
            del self.children[index]

            # shift class indices
            for i in range(index, len(self.children)):
                self.children[i].class_id -= 1

    def has_child(self, category_name: str) -> bool:
        """Return whether node has a child with name category_name"""
        return any(map(lambda c: c.category_name == category_name, self.children))

    @property 
    def num_children(self):
        return len(self.children)


#TODO add validity checks
class Taxonomy:
    def __init__(self, root: TaxonomyNode = None):
        # create generic root node if not passed
        self._root = (root 
                      if root is not None 
                      else TaxonomyNode(category_name="root", class_id=None))

    def __str__(self) -> str:
        """readable represenation of taxonomy"""
        return json.dumps(self._readable_repr(self._root), indent=4)

    def __contains__(self, category_name: str) -> bool:
        """Returns True if category name is in taxonomy else False"""
        return self.find_node_by_name(category_name)[0] is not None

    def add(self, parent_category: str, child_category: str):
        """
        Add child node to taxonomy, child's class id is assigned as index in parent's children
        """
        parent_node, _ = self.find_node_by_name(parent_category)

        if parent_node is None:
            raise ValueError(f"No node found with category: {parent_category}")

        # create child node and add to parent
        child_node = parent_node.add_child(child_category)
        return child_node

    def add_path(self, categories: List[str]):
        """
        Given a path of categories (L1, L2, ..., Lx) move to node if exists, else create.
        Allows easy creation of taxonomy given dataset with list of L1, ..., Lx per data point>
        e.g. If L1 exists, but L2 and L3 don't, then create L2 --> L1, and L3 --> L2

        :param categories: List of category names
        """
        pass

    def remove(self, category_name: str):
        """Remove node in taxonomy with name category_name"""
        node, path = self.find_node_by_name(category_name)
        
        if node is None:
            raise ValueError(f"No node found with category: {category_name}")

        # if the node path has length one, the parent must be the root
        parent_node = self._root if len(path) == 1 else path[-2]
        parent_node.remove_child(node.category_name)
        

    def has_link(self, parent_category, child_category):
        """Find nodes in self which are not in other"""
        parent_node, _ = self.find_node_by_name(parent_category)

        # could not find parent node
        if parent_node is None:
            return False

        # check if parent has specified child
        return parent_node.has_child(child_category)

    def category_name_to_class_ids(self, category_name: str) -> List[int]:
        """
        Given a category name which uniquely identifies a category, return L1...Lx class ids
        e.g. Categories x, y are L1, L2 categories for L3 category z then
             return [class_id(x), class_id(y), class_id(z)]
        """
        # walk through tree to find node by name (unique identifier)
        node, node_path = self.find_node_by_name(category_name)

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
    
    def validate(self):
        """Ensure that there are no duplicate category names"""
        names_set = set()
        for node in self.iter():
            if node.category_name not in names_set:
                names_set.add(node.category_name)
            else:
                raise ValueError("Category {node.category_name} is duplicated in taxonomy")

    def iter(self):
        return self._iter(self._root, 0)

    def iter_level(self, level: int = 1):
        return self._iter_level(self._root, level)    

    def read(self, dir_path: str) -> 'Taxonomy':
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

    def _iter(self, node: TaxonomyNode, depth: int):
        """Iterate recursively and output all nodes in tree except root"""
        if depth != 0:
            yield node

        for child in node.children:
            yield from self._iter(child, depth + 1)

    def _iter_level(self, node: TaxonomyNode, level: int):
        """Iterate recursively and output all nodes at given depth"""
        if level == 0:
            yield node
        else:
            for child in node.children:
                yield from self._iter_level(child, level - 1)

    def find_node_by_name(self, category_name: str) -> Tuple[TaxonomyNode, List[TaxonomyNode]]:
        return self._find_node_by_name(self._root, category_name, [])

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
            node, path = self._find_node_by_name(child, category_name, node_path)
            if node is not None:
                return node, path
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
