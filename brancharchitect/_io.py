from brancharchitect.newick_parser import parse_newick
from brancharchitect.node import serialize_to_dict_iterative
from uuid import UUID
import json


class UUIDEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        return json.JSONEncoder.default(self, obj)


def read_newick(path):
    with open(path) as f:
        newick_string = f.read()
    print(newick_string)
    tree = parse_newick(newick_string)
    return tree


def write_json(tree, path):
    serialized_tree = tree.to_dict()
    with open(path, mode='w') as f:
        json.dump(serialized_tree, f, cls=UUIDEncoder)
