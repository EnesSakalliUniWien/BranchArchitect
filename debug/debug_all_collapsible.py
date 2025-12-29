#!/usr/bin/env python3
"""Debug script to check all_collapsible_splits vs subtree paths"""

import logging

logging.basicConfig(level=logging.WARNING)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)
from brancharchitect.tree_interpolation.subtree_paths.paths.transition_builder import (
    calculate_subtree_paths,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver


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

    # Get the lattice solution
    solver = LatticeSolver(t5, t6)
    solution, _ = solver.solve_iteratively()

    # Find the pivot edge that has the problem split
    for pivot_edge, subtree_solutions in solution.items():
        collapse_splits, expand_splits = (
            get_unique_splits_for_current_pivot_edge_subtree(t5, t6, pivot_edge)
        )

        if problem_split in expand_splits:
            print(f"\n=== Pivot edge containing problem split ===")
            print(
                f"Pivot edge: {list(pivot_edge.indices)[:10]}... ({len(pivot_edge.indices)} taxa)"
            )
            print(
                f"all_collapsible_splits (from get_unique_splits): {len(collapse_splits)}"
            )
            print(f"all_expand_splits (from get_unique_splits): {len(expand_splits)}")

            # Calculate subtree paths
            dest_paths, source_paths = calculate_subtree_paths(
                {pivot_edge: subtree_solutions},
                t6,
                t5,
            )

            # Get the paths for this pivot edge
            dest_paths_for_pivot = dest_paths.get(pivot_edge, {})
            source_paths_for_pivot = source_paths.get(pivot_edge, {})

            # Count total splits in subtree paths
            all_collapse_in_paths = PartitionSet(encoding=t5.taxa_encoding)
            all_expand_in_paths = PartitionSet(encoding=t5.taxa_encoding)

            for subtree, paths in source_paths_for_pivot.items():
                all_collapse_in_paths |= paths
            for subtree, paths in dest_paths_for_pivot.items():
                all_expand_in_paths |= paths

            print(f"\nSubtree paths:")
            print(f"  Total collapse splits in paths: {len(all_collapse_in_paths)}")
            print(f"  Total expand splits in paths: {len(all_expand_in_paths)}")

            # Check which incompatible splits are missing from paths
            incompatible_in_pivot = [
                s for s in incompatible_collapse if s in collapse_splits
            ]
            missing_from_paths = [
                s for s in incompatible_in_pivot if s not in all_collapse_in_paths
            ]

            print(
                f"\nIncompatible splits in this pivot edge: {len(incompatible_in_pivot)}"
            )
            print(f"Missing from subtree paths: {len(missing_from_paths)}")
            for s in missing_from_paths:
                print(f"  - {list(s.indices)}")

            # Check which subtree has the problem split in its expand path
            print(f"\nSubtree expand path assignments:")
            for subtree, paths in dest_paths_for_pivot.items():
                if problem_split in paths:
                    print(f"  Problem split assigned to: {list(subtree.indices)}")

                    # Check collapse path for this subtree
                    collapse_path = source_paths_for_pivot.get(
                        subtree, PartitionSet(encoding=t5.taxa_encoding)
                    )
                    print(
                        f"  Collapse path for this subtree: {len(collapse_path)} splits"
                    )

                    # Check if incompatible splits are in this subtree's collapse path
                    missing_incompatible = [
                        s for s in incompatible_in_pivot if s not in collapse_path
                    ]
                    print(f"  Missing incompatible splits: {len(missing_incompatible)}")

            break


if __name__ == "__main__":
    main()
