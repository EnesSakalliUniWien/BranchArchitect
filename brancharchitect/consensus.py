from typing import List, Dict
from brancharchitect.tree import Node
from brancharchitect.split import Partition
from collections import Counter


def sort_splits(splits):
    return sorted(splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)


def create_consensus_tree(trees: List[Node]) -> Node:
    tree = create_star_tree(trees[0]._order)
    observed_number_of_splits = collect_splits(trees)
    for split, frequency in sorted(
        observed_number_of_splits.items(), key=lambda x: (x[1], x[0]), reverse=True
    ):
        if frequency < 1.0:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree(trees: List[Node], threshold: float = 0.50) -> Node:
    tree = create_star_tree(trees[0]._order)
    splits = collect_splits(trees)
    for split, frequency in sort_splits(splits):
        if frequency <= threshold:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree_extended(trees: List[Node]):
    tree = create_star_tree(trees[0]._order)
    splits = collect_splits(trees)
    applied_splits = []

    for split, frequency in sorted(
        splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True
    ):
        if all(compatible(split, existing_split) for existing_split in applied_splits):
            apply_split_in_tree(split, tree)
            applied_splits.append(split)
    return tree


def collect_splits(trees: List[Node]) -> Dict[Partition, float]:
    taxa = len(trees[0]._order)
    total_trees = len(trees)
    counter = Counter()
    for tree in trees:
        splits_in_tree = set(tree.to_splits())
        for split in splits_in_tree:
            if len(split) > 1 and len(split) < taxa:
                counter[split] += 1
    # Calculate frequencies
    counter = {split: count / total_trees for split, count in counter.items()}
    return counter


def create_star_tree(taxons: List[str]) -> Node:
    tree: Node = Node(name="Root", split_indices=tuple(range(len(taxons))))

    for index, name in enumerate(taxons):
        child = Node(name=name, split_indices=(index,), length=1)
        tree.append_child(child)  # This already calls invalidate_current_order_cache()
    return tree


def compatible(split1: Partition, split2: Partition):
    s1, s2 = set(split1), set(split2)
    return (s1 <= s2) or (s1 >= s2) or len((s1 & s2)) == 0


def apply_split_in_tree(split: Partition, node: Node) -> None:
    # Base case: if the node already has the split, return
    if node.split_indices == split:
        return

    split_set = set(split)
    node_split_set = set(node.split_indices)

    # The split can be applied if it's a proper subset of the node's split indices
    if split_set < node_split_set:
        group_a_children = []  # Children entirely within the split
        group_b_children = []  # Children entirely outside the split
        overlapping_children = []  # Children overlapping both sides

        for child in node.children:
            child_split_set = set(child.split_indices)
            if child_split_set <= split_set:
                # Child is entirely within the split_set
                group_a_children.append(child)
            elif child_split_set <= node_split_set - split_set:
                # Child is entirely outside the split_set
                group_b_children.append(child)
            else:
                # Child overlaps both sides; need to recurse
                overlapping_children.append(child)

        # Apply the split only if both groups have at least one child
        if group_a_children and group_b_children:
            # Remove current children and prepare to add new ones
            node.children = overlapping_children

            # Handle group A (inside the split)
            if len(group_a_children) == 1:
                # Only one child; attach directly without creating a new node
                node.children.append(group_a_children[0])
            else:
                # Create a new node for group A
                new_node_a = Node(
                    name="",
                    split_indices=tuple(sorted(split)),
                    children=group_a_children,
                )
                node.children.append(new_node_a)

            # Handle group B (outside the split)
            if len(group_b_children) == 1:
                # Only one child; attach directly without creating a new node
                node.children.append(group_b_children[0])
            else:
                # Create a new node for group B
                complement_split = tuple(sorted(node_split_set - split_set))
                new_node_b = Node(
                    name="",
                    split_indices=complement_split,
                    children=group_b_children,
                )
                node.children.append(new_node_b)

            # Sort the children for consistent ordering
            node.children = sorted(node.children, key=lambda n: n.split_indices)

            # Recurse into overlapping children
            for child in overlapping_children:
                apply_split_in_tree(split, child)

        else:
            # Recurse into children if the split wasn't applied here
            for child in node.children:
                apply_split_in_tree(split, child)
    else:
        # Recurse into the children
        for child in node.children:
            apply_split_in_tree(split, child)
