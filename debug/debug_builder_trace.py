#!/usr/bin/env python3
"""Debug script to trace builder execution"""

import logging

logging.basicConfig(level=logging.DEBUG)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)
from brancharchitect.tree_interpolation.subtree_paths.paths.transition_builder import (
    calculate_subtree_paths,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
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

    # Get the lattice solution
    solver = LatticeSolver(t5, t6)
    solution, _ = solver.solve_iteratively()

    # Find the pivot edge that has the problem split
    problem_indices = (15, 16, 17, 23, 27)

    for pivot_edge, subtree_solutions in solution.items():
        collapse_splits, expand_splits = (
            get_unique_splits_for_current_pivot_edge_subtree(t5, t6, pivot_edge)
        )

        # Check if problem split is in expand_splits
        problem_split = None
        for s in expand_splits:
            if set(s.indices) == set(problem_indices):
                problem_split = s
                break

        if problem_split:
            print(f"\n=== Found pivot edge ===")
            print(f"Pivot edge: {list(pivot_edge.indices)[:10]}...")
            print(f"all_collapsible_splits: {len(collapse_splits)}")
            print(f"all_expand_splits: {len(expand_splits)}")

            # Calculate subtree paths
            dest_paths, source_paths = calculate_subtree_paths(
                {pivot_edge: subtree_solutions},
                t6,
                t5,
            )

            dest_paths_for_pivot = dest_paths.get(pivot_edge, {})
            source_paths_for_pivot = source_paths.get(pivot_edge, {})

            print(f"\n=== Calling build_edge_plan ===")
            plans = build_edge_plan(
                dest_paths_for_pivot,
                source_paths_for_pivot,
                t5,
                t6,
                pivot_edge,
            )

            print(f"\n=== Plans ===")
            for subtree, plan in plans.items():
                collapse_path = plan["collapse"]["path_segment"]
                expand_path = plan["expand"]["path_segment"]
                print(f"\nSubtree: {list(subtree.indices)[:5]}...")
                print(f"  Collapse path: {len(collapse_path)} splits")
                print(f"  Expand path: {len(expand_path)} splits")

                if problem_split in expand_path:
                    print(f"  *** Problem split IS in expand path ***")

            break


if __name__ == "__main__":
    main()
