from brancharchitect.newick_parser import parse_newick
from brancharchitect.tree import serialize_to_dict_iterative, Node
from brancharchitect.svg import generate_svg
from uuid import UUID
import json


class UUIDEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        return json.JSONEncoder.default(self, obj)


def dump_json(tree, f):
    json.dump(tree, f, cls=UUIDEncoder)


def read_newick(path, order=None, force_list=False):
    with open(path) as f:
        newick_string = f.read()
    tree = parse_newick(newick_string, order=order, force_list=force_list)
    return tree


def write_json(tree, path):
    serialized_tree = tree.to_dict()
    with open(path, mode='w') as f:
        dump_json(serialized_tree, f)


def write_svg(tree, path):
    svg = generate_svg(tree)
    with open(path, mode='wb') as f:
        f.write(svg)


def serialize_tree_list_to_json(tree_list: list[Node]):
    serialized_tree_list = []
    for tree in tree_list:
        serialized_tree_list.append(tree.serialize_to_dict())
    return serialized_tree_list


def write_tree_dictionaries_to_json(tree_list: list[Node], file_name: str):
    serialized_tree_list = serialize_tree_list_to_json(tree_list)
    with open(file_name, "w") as f:
        dump_json(serialized_tree_list, f)

