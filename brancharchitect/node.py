import json


class Node:
    def __init__(self):
        self.children = []
        self.name = None
        self.indices = ""
        self.uuid = self.__hash__()
        self.length = None
        self.values = []
        self.split_indices = []
        self.parent = None
        self.leaf_name = None

    def append_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"Node({self.name} {self.indices})"

    def deep_copy(self):
        new_node = Node()
        new_node.name = self.name
        new_node.indices = self.indices
        new_node.uuid = (
            self.uuid
        )  # Depending on whether you want a unique uuid, you might want to generate a new one
        new_node.length = self.length

        # Deep copy the lists by creating a new list and copying each element
        new_node.values = [value for value in self.values]
        new_node.split_indices = [index for index in self.split_indices]

        for child in self.children:
            new_node.append_child(child.deep_copy())
        return new_node

    def serialize_to_dict(self):
        # Check if self.name is a set and convert it to a list if so
        name = list(self.name) if isinstance(self.name, set) else self.name

        serialized_node = {
            "name": name,
            "indices": self.indices,
            "uuid": self.uuid,
            "length": self.length,
            "values": self.values,
            "split_indices": self.split_indices,
            "leaf_name": self.leaf_name,
            "children": [],
        }

        for child in self.children:
            serialized_node["children"].append(child.serialize_to_dict())

        return serialized_node
    

    def dict_to_json_string(self):
        """
        Converts a dictionary representation of a node to a JSON string.
        :param serialized_node: The dictionary representation of the node.
        :return: JSON string representation of the node.
        """
        serialized_dict = self.serialize_to_dict()
        return json.dumps(serialized_dict, indent=4)


def serialize_to_dict_iterative(root):
    if root is None:
        return None

    stack = [(root, None)]  # Stack of tuples (node, parent_serialized_node)
    root_serialized = None

    while stack:
        node, parent_serialized = stack.pop()

        # Serialize current node
        serialized_node = {
            "name": list(node.name) if isinstance(node.name, set) else node.name,
            "indices": node.indices,
            "uuid": node.uuid,
            "length": node.length,
            "values": node.values,
            "split_indices": node.split_indices,
            "leaf_name": node.leaf_name,
            "children": []
        }

        # Attach to parent
        if parent_serialized is not None:
            parent_serialized["children"].append(serialized_node)
        else:
            root_serialized = serialized_node

        # Add children to stack
        for child in reversed(node.children):  # Reverse to maintain order
            stack.append((child, serialized_node))

    return root_serialized
