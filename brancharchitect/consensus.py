from typing import List, Dict, Tuple
from pprint import pprint
from brancharchitect.tree import Node, SplitIndices
from collections import Counter

def sort_splits(splits):
    return sorted(splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)

def create_consensus_tree(trees: List[Node]) -> Node:
    tree = create_star_tree(trees[0]._order)
    observed_number_of_splits = collect_splits(trees)    
    for split, frequency in sorted(observed_number_of_splits.items(), key=lambda x: (x[1], x[0]), reverse=True):
        if frequency < 1.0:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree(trees: List[Node]) -> Node:
    tree = create_star_tree(trees[0]._order)
    splits = collect_splits(trees)    
    for split, frequency in sort_splits(splits):
        if frequency <= 0.5:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree_extended(trees : List[Node]):
    tree = create_star_tree(trees[0]._order)
    splits = collect_splits(trees)
    applied_splits = []

    for split, frequency in sorted(splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True):
        if all(compatible(split, existing_split) for existing_split in applied_splits):
            apply_split_in_tree(split, tree)
            applied_splits.append(split)

    return tree


def collect_splits(trees: List[Node]) -> Dict[SplitIndices, float]:
    taxa = len(trees[0]._order)
    counter = Counter(split for tree in trees for split in tree.to_splits())
    counter = {split : count / len(trees) for split, count in counter.items() if len(split) > 1 and len(split) < taxa}
    return counter


def create_star_tree(taxons: List[str])-> Node:
    tree : Node = Node(name='Root', split_indices=tuple(range(len(taxons))))

    for index, name in enumerate(taxons):
        child = Node(name=name, split_indices=(index,), length=1)
        tree.append_child(child)
    return tree


def compatible(split1: SplitIndices, split2: SplitIndices):
    s1, s2 = set(split1), set(split2)
    return (s1 <= s2) or (s1 >= s2) or len((s1 & s2)) == 0


def apply_split_in_tree(split: SplitIndices, node: Node) -> Node:
    split = set(split)
    node_split = set(node.split_indices)
    if node_split < split or node_split > split:
        
        remaining_children, reassigned_children = [], []
        
        for child in node.children:
            child_split = set(child.split_indices)
            if child_split <= split or child_split >= split:
                reassigned_children.append(child)
            else:
                remaining_children.append(child)

        if reassigned_children:
            new_node = Node(name='', split_indices=tuple(sorted(split)), children=reassigned_children)
            # Update original node's children and append new node
            node.children = remaining_children        
            node.children.append(new_node)
            node.children = sorted(node.children, key=lambda node: node.split_indices)

    for child in node.children:
        if child.children:
            apply_split_in_tree(split, child)
