#!/usr/bin/env python3
"""Debug script to trace collapse paths for each subtree"""

import logging

logging.basicConfig(level=logging.WARNING)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
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
    unique_t5 = t5_splits - t6_splits
    unique_t6 = t6_splits - t5_splits

    # Problem split
    problem_indices = (15, 16, 17, 23, 27)
    problem_split = None
    for s in unique_t6:
        if set(s.indices) == set(problem_indices):
            problem_split = s
            break

    print(f"Problem split: {list(problem_split.indices)}")

    # Find incompatible collapse splits
    all_indices = set(t5.taxa_encoding.values())
    incompatible_collapse = []
    for s in unique_t5:
        if not problem_split.is_compatible_with(s, all_indices):
            incompatible_collapse.append(s)

    print(f"Incompatible collapse splits: {len(incompatible_collapse)}")
    for s in incompatible_collapse:
        print(f"  - {list(s.indices)}")

    # Use the root as pivot edge (all taxa)
    root_split = Partition(tuple(range(len(t5.taxa_encoding))), t5.taxa_encoding)

    collapse_splits, expand_splits = get_unique_splits_for_current_pivot_edge_subtree(
        t5, t6, root_split
    )

    print(f"\n=== Root pivot edge ===")
    print(f"Collapse splits: {len(collapse_splits)}")
    print(f"Expand splits: {len(expand_splits)}")

    # Check which incompatible splits are in this pivot edge's collapse splits
    incompatible_in_pivot = [s for s in incompatible_collapse if s in collapse_splits]
    print(f"\nIncompatible splits in this pivot edge: {len(incompatible_in_pivot)}")
    for s in incompatible_in_pivot:
        print(f"  - {list(s.indices)}")

    # Use simple subtree assignments - all to root
    collapse_by_subtree = {root_split: collapse_splits}
    expand_by_subtree = {root_split: expand_splits}

    print(f"\n=== Building edge plan (simple) ===")
    plans = build_edge_plan(
        expand_by_subtree,
        collapse_by_subtree,
        t5,
        t6,
        root_split,
    )

    for subtree, plan in plans.items():
        collapse_path = plan["collapse"]["path_segment"]
        expand_path = plan["expand"]["path_segment"]

        print(
            f"\nSubtree: {list(subtree.indices)[:5]}... ({len(subtree.indices)} taxa)"
        )
        print(f"  Collapse path: {len(collapse_path)} splits")
        print(f"  Expand path: {len(expand_path)} splits")

        if problem_split in expand_path:
            print(f"  Problem split IS in expand path")
            collapse_set = set(collapse_path)
            missing = [s for s in incompatible_in_pivot if s not in collapse_set]
            if missing:
                print(
                    f"  MISSING incompatible splits from collapse path: {len(missing)}"
                )
                for s in missing:
                    print(f"    - {list(s.indices)}")
            else:
                print(f"  All incompatible splits ARE in collapse path - GOOD!")


if __name__ == "__main__":
    main()
