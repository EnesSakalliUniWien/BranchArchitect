#!/usr/bin/env python3
"""Debug script to check which splits are being preserved during collapse"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition


def parse_newick_file(filepath):
    """Parse a newick file with multiple trees."""
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

    # Problem split
    problem_indices = (15, 16, 17, 23, 27)
    problem_split = None
    for s in unique_t6:
        if set(s.indices) == set(problem_indices):
            problem_split = s
            break

    print(f"Problem split: {list(problem_split.indices)}")
    print(f"Taxa: {list(problem_split.taxa)}")

    # Find incompatible collapse splits
    all_indices = set(t5.taxa_encoding.values())
    incompatible_collapse = []
    for s in unique_t5:
        if not problem_split.is_compatible_with(s, all_indices):
            incompatible_collapse.append(s)

    print(f"\nIncompatible collapse splits: {len(incompatible_collapse)}")
    for s in incompatible_collapse:
        print(f"  - {list(s.indices)}")
        # Check if this split exists in destination (T6)
        if s in t6_splits:
            print(f"    EXISTS in destination (T6) - WILL BE PRESERVED!")
        else:
            print(f"    NOT in destination (T6) - should be collapsed")

    # Check if any incompatible splits exist in destination
    preserved_incompatible = [s for s in incompatible_collapse if s in t6_splits]
    print(
        f"\nIncompatible splits that exist in destination: {len(preserved_incompatible)}"
    )

    if preserved_incompatible:
        print("BUG: These splits will be preserved but are incompatible with expand!")
    else:
        print("OK: No incompatible splits exist in destination")


if __name__ == "__main__":
    main()
