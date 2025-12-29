#!/usr/bin/env python3
"""Debug the path calculation to see if subtree is kept."""

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


def load_trees(filepath: str) -> list[Node]:
    with open(filepath, "r") as f:
        content = f.read()
    result = parse_newick(content, force_list=True)
    return result if isinstance(result, list) else [result]


def main():
    filepath = "datasets/all_trees copy 5.nwk"
    trees = load_trees(filepath)

    source_tree = trees[11]
    destination_tree = trees[12]
    destination_tree.initialize_split_indices(source_tree.taxa_encoding)

    jumping_subtree_solutions, _ = LatticeSolver(
        source_tree, destination_tree
    ).solve_iteratively()
    pivot_edges = list(jumping_subtree_solutions.keys())
    pivot_edge = pivot_edges[0]
    subtrees_for_pivot = jumping_subtree_solutions.get(pivot_edge, [])

    # Find the (146, 149) subtree
    target_subtree = None
    for subtree in subtrees_for_pivot:
        if tuple(subtree.indices) == (146, 149):
            target_subtree = subtree
            break

    if target_subtree:
        print(f"Found subtree (146, 149)")

        # Calculate paths using the function
        destination_subtree_paths, source_subtree_paths = calculate_subtree_paths(
            jumping_subtree_solutions, destination_tree, source_tree
        )

        dest_paths_for_pivot = destination_subtree_paths.get(pivot_edge, {})
        source_paths_for_pivot = source_subtree_paths.get(pivot_edge, {})

        dest_path = dest_paths_for_pivot.get(target_subtree, PartitionSet(encoding={}))
        source_path = source_paths_for_pivot.get(
            target_subtree, PartitionSet(encoding={})
        )

        print(f"\nAfter calculate_subtree_paths:")
        print(f"  Source path has {len(source_path)} splits")
        print(f"  Dest path has {len(dest_path)} splits")

        print(f"\n  Subtree in dest path: {target_subtree in dest_path}")

        if target_subtree in dest_path:
            print("  SUCCESS: Subtree is now kept in dest path!")
        else:
            print("  FAIL: Subtree is still being discarded!")


if __name__ == "__main__":
    main()
