import json
from dataclasses import dataclass, field, asdict
from uuid import uuid4

from copy import deepcopy

from typing import Optional, Any

@dataclass
class Node:

    children: list['Node'] = field(default_factory=list)
    name: Optional[str] = None
    indices: str = ''
    uuid: str = field(default_factory=uuid4)
    length: Optional[float] = None
    values: list[Any] = field(default_factory=list)
    split_indices: list[str] = field(default_factory=list)
    parent: Optional['Node'] = None
    leaf_name: Optional[str] = None

    def append_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"Node('{self.name}')"

    def deep_copy(self):
        return deepcopy(self)

    def to_dict(self):
        self._set_parent_none()
        return asdict(self)

    def _set_parent_none(self):
        self.parent = None
        for child in self.children:
            child._set_parent_none()

    def to_json(self):
        """
        Converts a dictionary representation of a node to a JSON string.
        :param serialized_node: The dictionary representation of the node.
        :return: JSON string representation of the node.
        """
        serialized_dict = self.serialize_to_dict()
        return json.dumps(serialized_dict, indent=4)


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
