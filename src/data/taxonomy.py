import json
import csv

from typing import List, Dict, Optional, Tuple
from os.path import join


class TaxonomyNode:
    """Represents a category in the taxonomy"""
    def __init__(self, 
                 category_id: int,
                 category_name: str, 
                 class_id: Optional[int], 
                 parent: 'TaxonomyNode' = None,
                 children: List['TaxonomyNode'] = None):
        """
        :param class_id: Class id for category, None for root
        """
        self._category_id = category_id
        self._category_name = category_name
        self._class_id = class_id
        self._parent = parent
        self._children = children if children is not None else []

    def add_child(self, category_id: int, category_name: str, class_id: Optional[int]):
        """
        Add child node to taxonomy node with name category_name

        Class id is auto-assigned as new index in self.children if not passed
        """
        child = TaxonomyNode(category_id=category_id,
                             category_name=category_name,
                             class_id=class_id if class_id is not None else len(self.children),
                             parent=self)

        # add child node and return it
        self.children.append(child)
        return child
    
    def remove_child(self, category_id: int):
        """
        Remove child node from taxonomy with id category_id

        Class ids of children will be shifted to stay consecutive
        """
        matches = [i for i, _ in enumerate(self._children) 
                   if c._category_id = category_id]

        if len(matches) == 0:
            raise ValueError(f"Node with id {category_id} not found")

        # id must be unique
        assert len(matches) == 1

        # remove child
        index = matches[0]
        del self.children[index]

        # shift class indices
        for i in range(index, len(self.children)):
            self.children[i].class_id -= 1

    def has_child(self, category_id: int) -> bool:
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
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent


class Taxonomy:
    def __init__(self, root: TaxonomyNode = None):
        # create generic root node if not passed
        self._root = (root 
                      if root is not None 
                      else TaxonomyNode(category_name="root", class_id=None))

    def __str__(self) -> str:
        """readable represenation of taxonomy"""
        return json.dumps(self._repr(), indent=4)

    def __contains__(self, category_name: str) -> bool:
        """Returns True if category name is in taxonomy else False"""
        return self.find_node_by_name(category_name)[0] is not None

    def iter(self):
        return self._iter(self._root, [])

    def iter_level(self, level: int = 1):
        return self._iter_level(self._root, level)    

    def get_max_level():
        """
        Return max level (L1, L2, etc.)
        Equivalent to the max depth - 1 (root)
        """
        return max(self.iter(), lambda x: len(x[1]))

    def add(self, parent_category_id: id, category_id: int, category_name: str, class_id: Optional[int] = None):
        """
        Add child node to taxonomy, child's class id is assigned as index in parent's children
        """
        parent_node, _ = self.find_node_by_id(parent_category_id)

        if parent_node is None:
            raise ValueError(f"No node found with category: {parent_category}")

        # create child node and add to parent
        child_node = parent_node.add_child(category_id, category_name, class_id)
        return child_node

    def remove(self, category_id: str) -> None:
        """Remove node in taxonomy with id category_id"""
        node, path = self.find_node_by_id(category_id)
        
        if node is None:
            raise ValueError(f"No node found with category id: {category_id}")

        # if the node path has length one, the parent must be the root
        parent_node = path[-2]
        parent_node.remove_child(node.category_id)
        
    def has_link(self, parent_category_id: int, child_category_id: int) -> bool:
        """Find nodes in self which are not in other"""
        parent_node, _ = self.find_node_by_id(parent_category_id)

        # could not find parent node
        if parent_node is None:
            return False

        # check if parent has specified child
        return parent_node.has_child(child_category_id)

    def find_node_by_id(self, category_id: int) -> Tuple[TaxonomyNode, List[TaxonomyNode]]:
        """
        Search through taxonomy to find node with category id category_id
        return: Tuple (node, path) where path is the list of nodes from root to desired node.
                Root is not included in node path 
        """
        for node, path in self.iter():
            if node.category_id == category_id:
                return node, path

        return None, None

    def read(self, dir_path: str) -> 'Taxonomy':
        """Read the human readable representation into native data structure"""
        # read taxonomy from dir
        taxonomy = Taxonomy()
        with open(join(dir_path, "taxonomy.csv"), 'r') as f:
            csv_reader = csv.DictReader(f)

            # parse each row
            for row in csv_reader:
                # compute depth of row
                depth = 1
                while f"L{depth}" in row and row[f"L{depth}"] != "":
                    depth += 1
                depth -= 1

                # extract node information 
                parent_id = row[f"L{depth - 1} Category ID"]
                category_id = row[f"L{depth} Category ID"]
                category_name = row[f"L{depth}"]
                class_id = row["Class ID"] if "Class ID" in row else None

                # add node to taxonomy
                taxonomy.add(parent_id, category_id, category_name, class_id)

        # parse into native data structure
        return Taxonomy(root=self._read_from_readable_repr(readable_taxonomy))
 
    def write(self, dir_path: str):
        """Write a human readable representation of taxonomy to a file in specified dir"""
        # write to a specific file name in dir
        with open(join(dir_path, "taxonomy.csv"), 'w') as f:
            # compute max depth
            max_level = self.get_max_level()

            header = [f"L{x}" for x in range(max_level)] 
                + [f"L{x} Category ID" for x in range(max_level)] 
                + ["Category ID", "Class ID"]

            csv_writer = csv.DictWriter(f, fieldnames=header)

            # header L1, ..., Lx, Category ID, Class ID
            csv_writer.writerow( + ["Category ID", "Class ID"])
            
            # write each path from root to node
            for names, ids, category_id, class_id in self._repr():
                csv_writer.writerow(names + [category_id, class_id])

    def _iter(self, node: TaxonomyNode, path: List[TaxonomyNode]):
        """Iterate recursively and output all nodes in tree except root"""
        yield node, path

        for child in node.children:
            path.append(child)
            yield from self._iter(child, path)
            path.pop()

    def _iter_level(self, node: TaxonomyNode, level: int):
        """Iterate recursively and output all nodes at given depth"""
        if level == 0:
            yield node
        else:
            for child in node.children:
                yield from self._iter_level(child, level - 1)

    def _repr() -> List[Tuple[List[str], Tuple[List[str]] int, int]]:
        """
        Return a representation of the taxonomy with one entry per node.
        Each entry is a list of names from root to node (excluding root), 
            a list of ids from root to node, id of the category, and class id.
        """
        rows = []
        for node, path in self.iter():
            # L1, ..., Lx category names and ids
            names = [n.category_name for n in (path[1:] + [node])]
            ids = [n.category_id for n in (path[1:] + [node])]

            # add Lx category id
            rows.append((names, ids, node.category_id, node.class_id))

        return rows
