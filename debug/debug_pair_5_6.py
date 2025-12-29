#!/usr/bin/env python3
"""Debug script to analyze pair 5->6 from 52_bootstrap.newick"""

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

    shared = t5_splits & t6_splits
    unique_t5 = t5_splits - t6_splits  # collapse
    unique_t6 = t6_splits - t5_splits  # expand

    print(f"Tree 5: {len(t5_splits)} splits")
    print(f"Tree 6: {len(t6_splits)} splits")
    print(f"Shared: {len(shared)}")
    print(f"Unique to T5 (collapse): {len(unique_t5)}")
    print(f"Unique to T6 (expand): {len(unique_t6)}")

    # Problem split
    problem_indices = (15, 16, 17, 23, 27)
    problem_split = Partition(problem_indices, t5.taxa_encoding)

    print(f"\nProblem split: {problem_indices}")
    print(f"Taxa: {list(problem_split.taxa)}")

    # Check what splits in the SHARED set are incompatible with problem split
    all_indices = set(t5.taxa_encoding.values())

    print("\n=== Checking incompatibilities ===")

    # Check against shared splits
    incompatible_shared = []
    for s in shared:
        if not problem_split.is_compatible_with(s, all_indices):
            incompatible_shared.append(s)

    print(f"\nIncompatible with SHARED splits: {len(incompatible_shared)}")
    for s in incompatible_shared:
        print(f"  - {list(s.indices)} = {list(s.taxa)}")

    # Check against unique_t5 (collapse) splits
    incompatible_collapse = []
    for s in unique_t5:
        if not problem_split.is_compatible_with(s, all_indices):
            incompatible_collapse.append(s)

    print(
        f"\nIncompatible with COLLAPSE (unique T5) splits: {len(incompatible_collapse)}"
    )
    for s in incompatible_collapse:
        print(f"  - {list(s.indices)} = {list(s.taxa)}")

    # Check against unique_t6 (expand) splits - should be 0 since they're all in the same tree
    incompatible_expand = []
    for s in unique_t6:
        if s != problem_split and not problem_split.is_compatible_with(s, all_indices):
            incompatible_expand.append(s)

    print(
        f"\nIncompatible with other EXPAND (unique T6) splits: {len(incompatible_expand)}"
    )
    for s in incompatible_expand:
        print(f"  - {list(s.indices)} = {list(s.taxa)}")

    # The key insight: after collapsing unique_t5, the tree should only have shared splits
    # Then we expand unique_t6 splits
    # If problem_split is incompatible with any shared split, that's a bug
    # If problem_split is incompatible with any collapse split that WASN'T collapsed, that's the bug

    print("\n=== Analysis ===")
    if incompatible_shared:
        print("BUG: Problem split is incompatible with SHARED splits!")
        print("This should be impossible since both trees are valid.")
    elif incompatible_collapse:
        print("These collapse splits MUST be removed before expanding problem split.")
        print("If they're not removed, the expand will fail.")
    else:
        print("No obvious incompatibilities found.")


if __name__ == "__main__":
    main()
