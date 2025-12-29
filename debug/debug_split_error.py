#!/usr/bin/env python3
"""Debug script to understand the SplitApplicationError with 52_bootstrap.newick"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)


def parse_newick_file(filepath):
    """Parse a newick file with multiple trees."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split by semicolon and filter empty strings
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
    # Load the trees
    trees = parse_newick_file("52_bootstrap.newick")
    print(f"Loaded {len(trees)} trees")

    # Try to process just the first 2 trees
    if len(trees) >= 2:
        t1 = trees[0]
        t2 = trees[1]

        print(f"\nTree 1 has {len(list(t1.to_splits()))} splits")
        print(f"Tree 2 has {len(list(t2.to_splits()))} splits")

        # Find shared and unique splits
        t1_splits = set(t1.to_splits())
        t2_splits = set(t2.to_splits())

        shared = t1_splits & t2_splits
        unique_t1 = t1_splits - t2_splits
        unique_t2 = t2_splits - t1_splits

        print(f"\nShared splits: {len(shared)}")
        print(f"Unique to T1 (collapse): {len(unique_t1)}")
        print(f"Unique to T2 (expand): {len(unique_t2)}")

        # Check for the problematic split
        problem_indices = (15, 16, 17, 23, 27)
        from brancharchitect.elements.partition import Partition

        problem_split = Partition(problem_indices, t1.taxa_encoding)

        print(f"\nProblem split {problem_indices}:")
        print(f"  In T1: {problem_split in t1_splits}")
        print(f"  In T2: {problem_split in t2_splits}")

        # Check compatibility with shared splits
        all_indices = set(t1.taxa_encoding.values())
        incompatible_with_shared = []
        for s in shared:
            if not problem_split.is_compatible_with(s, all_indices):
                incompatible_with_shared.append(s)

        print(
            f"\nProblem split incompatible with {len(incompatible_with_shared)} shared splits:"
        )
        for s in incompatible_with_shared[:5]:
            print(f"  - {list(s.indices)} = {list(s.taxa)}")

        # Check compatibility with unique_t1 splits
        incompatible_with_unique_t1 = []
        for s in unique_t1:
            if not problem_split.is_compatible_with(s, all_indices):
                incompatible_with_unique_t1.append(s)

        print(
            f"\nProblem split incompatible with {len(incompatible_with_unique_t1)} unique T1 splits:"
        )
        for s in incompatible_with_unique_t1[:5]:
            print(f"  - {list(s.indices)} = {list(s.taxa)}")

        # Try to run the pipeline with all trees
        print("\n\nAttempting pipeline processing with all trees...")
        try:
            pipeline = TreeInterpolationPipeline()
            pipeline.process_trees(trees=trees)
            print("Pipeline succeeded!")
        except Exception as e:
            print(f"Pipeline failed: {e}")

            # Try to find which pair fails
            print("\n\nTrying to find which pair fails...")
            for i in range(len(trees) - 1):
                try:
                    pipeline = TreeInterpolationPipeline()
                    pipeline.process_trees(trees=[trees[i], trees[i + 1]])
                except Exception as pair_e:
                    print(f"Pair {i} -> {i + 1} failed: {pair_e}")

                    # Analyze this pair
                    t1 = trees[i]
                    t2 = trees[i + 1]
                    t1_splits = set(t1.to_splits())
                    t2_splits = set(t2.to_splits())

                    print(f"\nTree {i} has {len(t1_splits)} splits")
                    print(f"Tree {i + 1} has {len(t2_splits)} splits")

                    shared = t1_splits & t2_splits
                    unique_t1 = t1_splits - t2_splits
                    unique_t2 = t2_splits - t1_splits

                    print(
                        f"Shared: {len(shared)}, Unique T1: {len(unique_t1)}, Unique T2: {len(unique_t2)}"
                    )

                    # Check if problem split is in either tree
                    problem_split = Partition(problem_indices, t1.taxa_encoding)
                    print(f"Problem split in T{i}: {problem_split in t1_splits}")
                    print(f"Problem split in T{i + 1}: {problem_split in t2_splits}")
                    break


if __name__ == "__main__":
    main()
