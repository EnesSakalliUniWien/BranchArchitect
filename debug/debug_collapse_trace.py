#!/usr/bin/env python3
"""Debug script to trace collapse execution"""

import logging

logging.basicConfig(level=logging.DEBUG)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.topology_ops.weights import (
    apply_zero_branch_lengths,
)
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
)


def parse_newick_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    tree_strings = [s.strip() + ";" for s in content.split(";") if s.strip()]
    trees = []
    order = None
    for tree_str in tree_strings:
        tree = parse_newick(tree_str, order)
        if order is None:
            order = list(tree.get_current_order())
        trees.append(tree)
    return trees


def main():
    trees = parse_newick_file("52_bootstrap.newick")
    t5 = trees[5]
    t6 = trees[6]

    t5_splits = set(t5.to_splits())
    t6_splits = set(t6.to_splits())
    unique_t5 = t5_splits - t6_splits  # collapse
    unique_t6 = t6_splits - t5_splits  # expand

    print(f"T5 splits: {len(t5_splits)}")
    print(f"T6 splits: {len(t6_splits)}")
    print(f"Unique to T5 (collapse): {len(unique_t5)}")
    print(f"Unique to T6 (expand): {len(unique_t6)}")

    # Problem split
    problem_indices = (15, 16, 17, 23, 27)
    problem_split = None
    for s in unique_t6:
        if set(s.indices) == set(problem_indices):
            problem_split = s
            break

    print(f"\nProblem split: {list(problem_split.indices)}")

    # Find incompatible collapse splits
    all_indices = set(t5.taxa_encoding.values())
    incompatible_collapse = []
    for s in unique_t5:
        if not problem_split.is_compatible_with(s, all_indices):
            incompatible_collapse.append(s)

    print(f"Incompatible collapse splits: {len(incompatible_collapse)}")
    for s in incompatible_collapse:
        print(f"  - {list(s.indices)}")

    # Simulate the collapse phase
    print(f"\n=== Simulating collapse phase ===")

    # Create a copy of T5
    tree = t5.deep_copy()

    # Get the root split (pivot edge)
    root_split = Partition(tuple(range(len(t5.taxa_encoding))), t5.taxa_encoding)

    # Apply zero branch lengths to all collapse splits
    print(f"\nApplying zero branch lengths to {len(unique_t5)} collapse splits...")
    it_down = apply_zero_branch_lengths(tree, PartitionSet(unique_t5))

    # Check which splits have zero length
    zero_length_splits = []
    for split in unique_t5:
        node = it_down.find_node_by_split(split)
        if node is not None:
            if node.length is not None and node.length <= 0:
                zero_length_splits.append(split)

    print(f"Splits with zero length: {len(zero_length_splits)}")

    # Check if incompatible splits have zero length
    incompatible_with_zero = [
        s for s in incompatible_collapse if s in zero_length_splits
    ]
    print(f"Incompatible splits with zero length: {len(incompatible_with_zero)}")

    # Now collapse
    print(f"\nCollapsing zero-length branches...")
    collapsed = create_collapsed_consensus_tree(
        it_down, root_split, destination_tree=t6
    )

    collapsed_splits = set(collapsed.to_splits())
    print(f"Collapsed tree splits: {len(collapsed_splits)}")

    # Check if incompatible splits are still in the tree
    remaining_incompatible = [s for s in incompatible_collapse if s in collapsed_splits]
    print(f"Remaining incompatible splits: {len(remaining_incompatible)}")
    for s in remaining_incompatible:
        print(f"  - {list(s.indices)}")
        # Check if this split exists in destination
        if s in t6_splits:
            print(f"    EXISTS in destination - preserved correctly")
        else:
            print(f"    NOT in destination - BUG: should have been collapsed!")


if __name__ == "__main__":
    main()
