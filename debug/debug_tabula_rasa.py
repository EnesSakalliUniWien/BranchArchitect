#!/usr/bin/env python3
"""Debug script to trace TABULA RASA collapse path"""

import logging

logging.basicConfig(level=logging.DEBUG)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.tree_interpolation.subtree_paths.planning.subtree_path_builder import (
    build_subtree_paths,
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
            print(f"Collapse splits: {len(collapse_splits)}")
            print(f"Expand splits: {len(expand_splits)}")

            # Build subtree paths
            collapse_by_subtree, expand_by_subtree = build_subtree_paths(
                subtree_solutions,
                collapse_splits,
                expand_splits,
                pivot_edge,
            )

            print(f"\n=== Subtree path assignments ===")
            for subtree, splits in collapse_by_subtree.items():
                print(
                    f"Collapse - {list(subtree.indices)[:5]}...: {len(splits)} splits"
                )
            for subtree, splits in expand_by_subtree.items():
                print(f"Expand - {list(subtree.indices)[:5]}...: {len(splits)} splits")

            # Initialize state
            state = PivotSplitRegistry(
                collapse_splits,
                expand_splits,
                collapse_by_subtree,
                expand_by_subtree,
                pivot_edge,
            )

            print(f"\n=== PivotSplitRegistry state ===")
            print(f"all_collapsible_splits: {len(state.all_collapsible_splits)}")
            print(f"first_subtree_processed: {state.first_subtree_processed}")

            tabula_rasa = state.get_tabula_rasa_collapse_splits()
            print(f"tabula_rasa collapse splits: {len(tabula_rasa)}")

            # Check if all incompatible splits are in tabula_rasa
            incompatible_in_pivot = [
                s for s in incompatible_collapse if s in collapse_splits
            ]
            missing = [s for s in incompatible_in_pivot if s not in tabula_rasa]
            if missing:
                print(f"\nMISSING incompatible splits from tabula_rasa: {len(missing)}")
                for s in missing:
                    print(f"  - {list(s.indices)}")
            else:
                print(f"\nAll incompatible splits ARE in tabula_rasa - GOOD!")

            break


if __name__ == "__main__":
    main()
