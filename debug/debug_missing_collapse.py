#!/usr/bin/env python3
"""Debug script to find which collapse splits are missing from the first subtree's collapse path"""

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
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

    # Get the lattice solution to understand subtree assignments
    solver = LatticeSolver(t5, t6)
    solution, _ = solver.solve_iteratively()

    print(f"\n=== Lattice Solution ===")
    print(f"Pivot edges: {len(solution)}")

    # Find the pivot edge that has the most collapse/expand splits
    # This is likely the one causing the issue
    for pivot_edge, subtree_solutions in solution.items():
        collapse_splits, expand_splits = (
            get_unique_splits_for_current_pivot_edge_subtree(t5, t6, pivot_edge)
        )

        if problem_split in expand_splits:
            print(f"\n=== Pivot edge containing problem split ===")
            print(
                f"Pivot edge: {list(pivot_edge.indices)[:10]}... ({len(pivot_edge.indices)} taxa)"
            )
            print(f"Collapse splits: {len(collapse_splits)}")
            print(f"Expand splits: {len(expand_splits)}")
            print(f"Subtree solutions: {len(subtree_solutions)}")

            # Check which incompatible splits are in this pivot edge
            incompatible_in_pivot = [
                s for s in incompatible_collapse if s in collapse_splits
            ]
            print(
                f"\nIncompatible splits in this pivot edge: {len(incompatible_in_pivot)}"
            )

            # Print subtree solutions
            print(f"\nSubtree solutions:")
            for subtree in subtree_solutions:
                print(
                    f"  - {list(subtree.indices)[:5]}... ({len(subtree.indices)} taxa)"
                )

            break


if __name__ == "__main__":
    main()
