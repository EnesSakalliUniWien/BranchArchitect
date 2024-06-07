from brancharchitect.newick_parser import parse_newick
from brancharchitect.node import Node
import json

__all__ = ['interpolate_tree', 'interpolate_adjacent_tree_pairs']

### public API ###

def interpolate_tree(tree_one: Node, tree_two: Node):
    split_dict1 = get_split_dict(tree_one)
    split_dict2 = get_split_dict(tree_two)

    it1 = calculate_intermediate_tree(tree_one, split_dict2)
    it2 = calculate_intermediate_tree(tree_two, split_dict1)

    c1 = calculate_consensus_tree(it1, split_dict2)
    c2 = calculate_consensus_tree(it2, split_dict1)

    return (it1, c1, c2, it2)


def interpolate_adjacent_tree_pairs(tree_list) -> list[Node]:
    results = []
    for i in range(len(tree_list) - 1):
        tree_one = tree_list[i]
        tree_two = tree_list[i+1]

        # Interpolate trees and get intermediate and consensus trees
        trees = interpolate_tree(tree_one, tree_two)

        results.append(tree_one)
        results.extend(trees)

    results.append(tree_two)
    return results


### Private API ###


def calculate_intermediate_tree(tree, split_dict):
    it = tree.deep_copy()
    _calculate_intermediate_tree(it, split_dict)
    return it


def _calculate_intermediate_tree(node, split_dict):
    if node.split_indices not in split_dict:
        node.length = 0
    else:
        node.length = (split_dict[node.split_indices] + node.length) / 2
    for child in node.children:
        _calculate_intermediate_tree(child, split_dict)


def calculate_consensus_tree(node, split_dict):
    node = node.deep_copy()
    return _calculate_consensus_tree(node, split_dict)


def _calculate_consensus_tree(node, split_dict):
    new_children = []
    for child in node.children:
        processed_child = _calculate_consensus_tree(child, split_dict)
        if processed_child.split_indices in split_dict:
            new_children.append(processed_child)
        else:
            for child in processed_child.children:
                new_children.append(child)

    node.children = new_children
    return node


def get_split_dict(node: Node, split_dict: dict[list[int], float]=None):
    if split_dict == None:
        split_dict = {}
    for child in node.children:
        split_dict = get_split_dict(child, split_dict)
    split_dict[node.split_indices] = node.length
    return split_dict
