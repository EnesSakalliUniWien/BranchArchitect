from typing import Dict, List, IO, Tuple
from xml.etree.ElementTree import Element
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.plot.circular_tree import (
    generate_multiple_circular_trees_svg,
)
from uuid import UUID
import json
from typing import Any, Optional


class UUIDEncoder(json.JSONEncoder):
    def default(self, o: Any):
        # Handle Partition objects
        if o.__class__.__name__ == "Partition":
            # Just return the indices as a list
            return list(o.indices)

        # Original UUID handling code
        if isinstance(o, UUID):
            return str(o)

        # Add handling for PartitionSet too
        if o.__class__.__name__ == "PartitionSet":
            # Return a list of lists of indices
            return [list(partition.indices) for partition in o]

        # Add handling for Node objects
        if o.__class__.__name__ == "Node":
            # Convert Node to a serializable dictionary
            node_dict = {
                "name": o.name,
                "length": o.length,
                "values": o.values,
                "children": o.children,  # This will recursively serialize child nodes
            }
            # Only include non-default/non-empty fields to keep JSON clean
            if hasattr(o, "split_indices") and o.split_indices:
                node_dict["split_indices"] = o.split_indices
            return node_dict

        return super().default(o)


def dump_json(tree: Node, f: IO[str]):
    json.dump(tree, f, cls=UUIDEncoder)


def read_newick(path: str, order: Optional[list[str]] = None, force_list: bool = False):
    with open(path) as f:
        newick_string: str = f.read()

    tree: Node | List[Node] = parse_newick(
        newick_string, order=order, force_list=force_list
    )
    return tree


def write_json(tree: Node, path: str):
    with open(path, mode="w") as f:
        dump_json(tree, f)


def write_svg(tree: Node, path: str, ignore_branch_lengths: bool = False):
    svg: Tuple[Element[str], List[Dict[str, Tuple[float, float]]]] = (
        generate_multiple_circular_trees_svg(
            [tree], ignore_branch_lengths=ignore_branch_lengths
        )
    )
    with open(path, mode="wb") as f:
        f.write(svg)


def serialize_tree_list_to_json(tree_list: List[Node]) -> List[Dict[str, Any]]:
    serialized_tree_list: List[Dict[str, Any]] = []
    for i, tree in enumerate(tree_list):
        d: Dict[str, Any] = tree.to_dict()
        if i == 0:
            print(
                f"[SERIALIZE DEBUG] type: {type(d['split_indices'])}, value: {d['split_indices']}"
            )
        serialized_tree_list.append(d)
    return serialized_tree_list


def write_tree_dictionaries_to_json(tree_list: list[Node], file_name: str):
    serialized_tree_list = serialize_tree_list_to_json(tree_list)
    with open(file_name, "w") as f:
        json.dump(serialized_tree_list, f, cls=UUIDEncoder)
