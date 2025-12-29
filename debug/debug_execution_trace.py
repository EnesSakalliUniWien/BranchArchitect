#!/usr/bin/env python3
"""Trace the actual execution to see where splits are lost."""

import logging
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.tree_interpolation.subtree_paths.paths import (
    calculate_subtree_paths,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.tree_interpolation.subtree_paths.execution.microsteps import (
    extract_filtered_paths,
)

logging.basicConfig(level=logging.WARNING)


def load_trees(filepath: str) -> list[Node]:
    with open(filepath, "r") as f:
        content = f.read()
    result = parse_newick(content, force_list=True)
    return result if isinstance(result, list) else [result]


def find_split_by_indices(
    splits: PartitionSet, target_indices: tuple
) -> Partition | None:
    for split in splits:
        if tuple(split.indices) == target_indices:
            return split
    return None


def main():
    filepath = "datasets/all_trees copy 5.nwk"
    trees = load_trees(filepath)

    source_tree = trees[11]
    destination_tree = trees[12]
    destination_tree.initialize_split_indices(source_tree.taxa_encoding)

    dest_splits = destination_tree.to_splits()

    missing_indices = [
        (356, 357, 358, 359, 365, 366, 368),
        (146, 149),
    ]

    missing_partitions = []
    for indices in missing_indices:
        split = find_split_by_indices(dest_splits, indices)
        if split:
            missing_partitions.append(split)

    jumping_subtree_solutions, _ = LatticeSolver(
        source_tree, destination_tree
    ).solve_iteratively()
    pivot_edges = list(jumping_subtree_solutions.keys())
    pivot_edge = pivot_edges[0]
    subtrees_for_pivot = jumping_subtree_solutions.get(pivot_edge, [])

    destination_subtree_paths, source_subtree_paths = calculate_subtree_paths(
        jumping_subtree_solutions, destination_tree, source_tree
    )
    dest_paths_for_pivot = destination_subtree_paths.get(pivot_edge, {})
    source_paths_for_pivot = source_subtree_paths.get(pivot_edge, {})

    # Build the plan
    plans = build_edge_plan(
        expand_splits_by_subtree=dest_paths_for_pivot,
        collapse_splits_by_subtree=source_paths_for_pivot,
        collapse_tree=source_tree,
        expand_tree=destination_tree,
        current_pivot_edge=pivot_edge,
    )

    # Find the subtrees that have the missing splits in their plans
    print("=== Tracing execution for missing splits ===")
    for missing in missing_partitions:
        print(f"\nMissing split {tuple(missing.indices)}:")
        missing_taxa = set(missing.taxa)

        for subtree, plan in plans.items():
            expand_path = plan.get("expand", {}).get("path_segment", [])
            if missing in expand_path:
                subtree_taxa = set(subtree.taxa)
                overlaps = subtree_taxa.intersection(missing_taxa)

                print(f"  In plan for subtree {tuple(subtree.indices)}")
                print(f"    Subtree taxa overlap with missing: {len(overlaps) > 0}")

                # Simulate the filtering in build_microsteps_for_selection
                selection = {
                    "subtree": subtree,
                    "expand": plan["expand"],
                    "collapse": plan.get("collapse", {}),
                }
                filtered_expand, _ = extract_filtered_paths(selection, pivot_edge)

                if missing in filtered_expand:
                    print(f"    After extract_filtered_paths: KEPT")

                    # Now simulate the mover_taxa filter
                    mover_taxa = set(subtree.taxa)
                    # Assume other_movers is not empty (which triggers the filter)
                    would_keep = bool(mover_taxa.intersection(set(missing.taxa)))
                    if would_keep:
                        print(f"    After mover_taxa filter: KEPT")
                    else:
                        print(f"    After mover_taxa filter: FILTERED OUT!")
                        print(f"    Subtree taxa: {list(subtree_taxa)[:3]}...")
                        print(f"    Missing taxa: {list(missing_taxa)[:3]}...")
                else:
                    print(f"    After extract_filtered_paths: FILTERED OUT")
                break  # Just check first matching subtree


if __name__ == "__main__":
    main()
