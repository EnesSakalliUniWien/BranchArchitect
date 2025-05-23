from brancharchitect.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.plot.circular_tree import (
    generate_multiple_circular_trees_svg,)
from uuid import UUID
import json


class UUIDEncoder(json.JSONEncoder):
    def default(self, o):
        # Handle Partition objects
        if o.__class__.__name__ == 'Partition':
            # Just return the indices as a list
            return list(o.indices)
        
        # Original UUID handling code
        if isinstance(o, UUID):
            return str(o)
            
        # Add handling for PartitionSet too
        if o.__class__.__name__ == 'PartitionSet':
            # Return a list of lists of indices
            return [list(partition.indices) for partition in o]
            
        return super().default(o)

def dump_json(tree, f):
    json.dump(tree, f, cls=UUIDEncoder)


def read_newick(path, order=None, force_list=False):
    with open(path) as f:
        newick_string = f.read()

    tree = parse_newick(newick_string, order=order, force_list=force_list)
    return tree


def write_json(tree, path):
    serialized_tree = tree.to_dict()
    with open(path, mode="w") as f:
        dump_json(serialized_tree, f)


def write_svg(tree, path, ignore_branch_lengths=False):
    svg = generate_multiple_circular_trees_svg(
        [tree], ignore_branch_lengths=ignore_branch_lengths
    )
    with open(path, mode="wb") as f:
        f.write(svg)


def serialize_tree_list_to_json(tree_list: list[Node]):
    serialized_tree_list = []
    for i, tree in enumerate(tree_list):
        d = tree.to_dict()
        if i == 0:
            print(f"[SERIALIZE DEBUG] type: {type(d['split_indices'])}, value: {d['split_indices']}")
        serialized_tree_list.append(d)
    return serialized_tree_list


def write_tree_dictionaries_to_json(tree_list: list[Node], file_name: str):
    serialized_tree_list = serialize_tree_list_to_json(tree_list)
    with open(file_name, "w") as f:
        dump_json(serialized_tree_list, f)
