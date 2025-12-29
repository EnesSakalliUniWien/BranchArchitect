#!/usr/bin/env python3
"""Debug script to trace the actual error in pair 5->6"""

import logging

logging.basicConfig(level=logging.DEBUG)

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)


def parse_newick_file(filepath):
    """Parse a newick file with multiple trees."""
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

    # Only process trees 5 and 6
    subset = trees[5:7]

    print(f"Processing {len(subset)} trees")
    print(f"Tree 5 splits: {len(subset[0].to_splits())}")
    print(f"Tree 6 splits: {len(subset[1].to_splits())}")

    pipeline = TreeInterpolationPipeline()

    try:
        result = pipeline.process_trees(trees=subset)
        print(f"Success! Generated {len(result.trees)} interpolation trees")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
