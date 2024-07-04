import json
from dataclasses import dataclass, field, asdict
from uuid import uuid4
from copy import deepcopy
from typing import Optional, Any

@dataclass()
class Node:

    children: list['Node'] = field(default_factory=list, compare=False)
    name: Optional[str] = field(default=None, compare=False)
    indices: str = field(default='', compare=False)
    uuid: str = field(default_factory=uuid4, compare=False)
    length: Optional[float] = field(default=None, compare=True)
    values: list[Any] = field(default_factory=list, compare=True)
    split_indices: list[str] = field(default_factory=list, compare=True)
    leaf_name: Optional[str] = field(default=None, compare=False)

    def append_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"Node('{self.name}')"

    def __hash__(self):
        return hash(str(self.uuid))

    def deep_copy(self):
        return deepcopy(self)

    def to_dict(self):
        return asdict(self)

    def to_newick(self):
        return self._to_newick() + ';'

    def _to_newick(self):
        length = ''
        if self.length is not None:
            length = f':{self.length}'

        meta = ''
        if self.values:
            meta = '[' + ','.join(f'{key}={value}' for key, value in self.values.items()) + ']'

        children = ''
        if self.children:
            if(len(self.children) > 1):            
                children = '(' + ','.join(child._to_newick() for child in self.children) + ')'
            else: 
                children =  ','.join(child._to_newick() for child in self.children)
        return f'{children}{self.name}{meta}{length}'

    def to_json(self):
        """
        Converts a dictionary representation of a node to a JSON string.
        :param serialized_node: The dictionary representation of the node.
        :return: JSON string representation of the node.
        """
        serialized_dict = self.to_dict()
        return json.dumps(serialized_dict, indent=4)

    def _initialize_split_indices(self, order):
        if len(self.split_indices) > 0:
            return
        for child in self.children:
            child._initialize_split_indices(order)
        if not self.children:
            self.split_indices = (order.index(self.name),)
        else:
            self.split_indices = []
            for child in self.children:
                self.split_indices.extend(child.split_indices)
            self.split_indices = tuple(sorted(self.split_indices))

            #if self. == '':
                #self.name = ','.join(str(s) for s in self.split_indices)



def serialize_to_dict_iterative(root):
    # DONE how does this differ from root.to_dict() ?
    # This differs by using an iterative approach (compared to a recursive approach), which does not run into the recursion depth limit
    # TODO change default implementation of Node.to_dict() to use an iterative approach, so that very deep trees can be handled.
    if root is None:
        return None

    stack = [(root, None)]  # Stack of tuples (node, parent_serialized_node)
    root_serialized = None

    while stack:
        node, parent_serialized = stack.pop()

        # Serialize current node
        serialized_node = node.to_dict()

        # Attach to parent
        if parent_serialized is not None:
            parent_serialized["children"].append(serialized_node)
        else:
            root_serialized = serialized_node

        # Add children to stack
        for child in reversed(node.children):  # Reverse to maintain order
            stack.append((child, serialized_node))

    return root_serialized
