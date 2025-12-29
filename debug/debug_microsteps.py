#!/usr/bin/env python3
"""Debug script to trace microsteps execution"""

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
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.tree_interpolation.subtree_paths.execution.microsteps import (
    build_microsteps_for_selection,
    extract_filtered_paths,
)
from brancharchitect.tree_interpolation.topology_ops.weights import (
    apply_zero_branch_lengths,
)
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
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

            # Calculate subtree paths
            dest_paths, source_paths = calculate_subtree_paths(
                {pivot_edge: subtree_solutions},
                t6,
                t5,
            )

            dest_paths_for_pivot = dest_paths.get(pivot_edge, {})
            source_paths_for_pivot = source_paths.get(pivot_edge, {})

            # Build edge plan
            plans = build_edge_plan(
                dest_paths_for_pivot,
                source_paths_for_pivot,
                t5,
                t6,
                pivot_edge,
            )

            # Simulate microsteps execution
            print(f"\n=== Simulating microsteps ===")

            interpolation_state = t5.deep_copy()

            for subtree, plan in plans.items():
                collapse_path = plan["collapse"]["path_segment"]
                expand_path = plan["expand"]["path_segment"]

                print(f"\nSubtree: {list(subtree.indices)[:5]}...")
                print(f"  Collapse path: {len(collapse_path)} splits")
                print(f"  Expand path: {len(expand_path)} splits")

                # Check if problem split is in expand path
                if problem_split in expand_path:
                    print(f"  *** Problem split IS in expand path ***")

                    # Check tree state before this subtree
                    current_splits = set(interpolation_state.to_splits())
                    remaining_incompatible = [
                        s for s in incompatible_collapse if s in current_splits
                    ]
                    print(
                        f"  Remaining incompatible splits in tree: {len(remaining_incompatible)}"
                    )
                    for s in remaining_incompatible:
                        print(f"    - {list(s.indices)}")

                    if remaining_incompatible:
                        print(f"  BUG: Incompatible splits should have been collapsed!")

                # Apply collapse
                if collapse_path:
                    selection = {
                        "subtree": subtree,
                        "collapse": {"path_segment": collapse_path},
                        "expand": {"path_segment": expand_path},
                    }

                    expand_path_filtered, zeroing_path = extract_filtered_paths(
                        selection, pivot_edge
                    )

                    print(f"  Zeroing path (filtered): {len(zeroing_path)} splits")

                    # Apply zero branch lengths
                    it_down = apply_zero_branch_lengths(
                        interpolation_state, PartitionSet(set(zeroing_path))
                    )

                    # Collapse
                    collapsed = create_collapsed_consensus_tree(
                        it_down, pivot_edge, destination_tree=t6
                    )

                    collapsed_splits = set(collapsed.to_splits())
                    print(f"  After collapse: {len(collapsed_splits)} splits")

                    # Check if incompatible splits were collapsed
                    remaining_after_collapse = [
                        s for s in incompatible_collapse if s in collapsed_splits
                    ]
                    print(
                        f"  Remaining incompatible after collapse: {len(remaining_after_collapse)}"
                    )

                    interpolation_state = collapsed

            break


if __name__ == "__main__":
    main()
