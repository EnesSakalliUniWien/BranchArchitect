#!/usr/bin/env python3
"""Debug why (146, 149) is not being created."""

from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)


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
    source_splits = source_tree.to_splits()

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
        print(f"  In source splits: {target_subtree in source_splits}")
        print(f"  In dest splits: {target_subtree in dest_splits}")

        # Check the path from subtree to pivot in both trees
        source_path = source_tree.find_path_between_splits(target_subtree, pivot_edge)
        dest_path = destination_tree.find_path_between_splits(
            target_subtree, pivot_edge
        )

        print(f"\n  Source path length: {len(source_path)}")
        print(f"  Dest path length: {len(dest_path)}")

        source_path_splits = [n.split_indices for n in source_path]
        dest_path_splits = [n.split_indices for n in dest_path]

        print(
            f"\n  Source path splits: {[tuple(s.indices) for s in source_path_splits]}"
        )
        print(f"  Dest path splits: {[tuple(s.indices) for s in dest_path_splits]}")

        # Check if the subtree itself is in the path
        print(f"\n  Subtree in source path: {target_subtree in source_path_splits}")
        print(f"  Subtree in dest path: {target_subtree in dest_path_splits}")

        # The issue: if subtree is in dest path but not source path, it needs to be created
        # But it gets discarded because it's the subtree endpoint
        if (
            target_subtree in dest_path_splits
            and target_subtree not in source_path_splits
        ):
            print("\n  *** ISSUE: Subtree is in dest path but not source path ***")
            print("  This means the subtree split needs to be CREATED")
            print("  But it gets discarded as the subtree endpoint!")


if __name__ == "__main__":
    main()
