from typing import Dict, List, IO, Any
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
import json
from typing import Optional
from brancharchitect.uuid_encoder import UUIDEncoder


def dump_json(tree: Node, f: IO[str]):
    json.dump(tree, f, cls=UUIDEncoder)


def read_newick(
    path: str,
    order: Optional[list[str]] = None,
    force_list: bool = False,
    treat_zero_as_epsilon: bool = False,
):
    with open(path) as f:
        newick_string: str = f.read()

    tree: Node | List[Node] = parse_newick(
        newick_string,
        order=order,
        force_list=force_list,
        treat_zero_as_epsilon=treat_zero_as_epsilon,
    )
    return tree


def write_json(tree: Node, path: str):
    with open(path, mode="w") as f:
        dump_json(tree, f)


def serialize_tree_list_to_json(tree_list: List[Node]) -> List[Dict[str, Any]]:
    serialized_tree_list: List[Dict[str, Any]] = []
    for tree in tree_list:
        d: Dict[str, Any] = tree.to_dict()
        serialized_tree_list.append(d)
    return serialized_tree_list


def write_tree_dictionaries_to_json(tree_list: list[Node], file_name: str):
    serialized_tree_list = serialize_tree_list_to_json(tree_list)
    with open(file_name, "w") as f:
        json.dump(serialized_tree_list, f, cls=UUIDEncoder)


def serialize_subtree_tracking(
    tracking: Optional[List[Optional[List[Partition]]]],
) -> List[Optional[List[List[int]]]]:
    """
    Serialize partition tracking to index arrays for JSON serialization.

    Converts each grouped list of Partitions to a list of sorted lists of integer indices.
    None values remain None.

    Args:
        tracking: List of Optional[List[Partition]] from the interpolation sequence

    Returns:
        List of Optional[List[List[int]]] suitable for JSON serialization
    """
    serialized: List[Optional[List[List[int]]]] = []
    if tracking is None:
        return serialized

    for group in tracking:
        if group is None:
            serialized.append(None)
        else:
            # group is List[Partition] -> convert to List[List[int]]
            # Ensure each partition's indices are sorted lists
            serialized.append([list(p.indices) for p in group])
    return serialized
