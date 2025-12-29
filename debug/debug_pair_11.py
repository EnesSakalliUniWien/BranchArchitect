#!/usr/bin/env python3
"""Debug script to trace where splits are being lost in pair 11."""

import logging
from pathlib import Path
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)

# Enable detailed logging for the orchestrator
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger(
    "brancharchitect.tree_interpolation.subtree_paths.planning.diagnostics"
).setLevel(logging.WARNING)


def load_trees(filepath: str) -> list[Node]:
    """Load trees from a Newick file."""
    with open(filepath, "r") as f:
        content = f.read()

    result = parse_newick(content, force_list=True)
    if isinstance(result, list):
        return result
    return [result]


def main():
    filepath = "datasets/all_trees copy 5.nwk"
    trees = load_trees(filepath)

    # Test pair 11 (trees 11 and 12)
    source_tree = trees[11]
    destination_tree = trees[12]

    # Ensure same encoding
    destination_tree.initialize_split_indices(source_tree.taxa_encoding)

    # Get the missing splits from destination
    dest_splits = destination_tree.to_splits()
    source_splits = source_tree.to_splits()

    # The missing splits from the error
    missing_indices = [
        (356, 357, 358, 359, 365, 366, 368),
        (146, 149),
        (253, 254, 259, 260, 261, 262, 263, 264),
    ]

    print("=== Checking missing splits ===")
    for indices in missing_indices:
        # Find the split in destination
        for split in dest_splits:
            if tuple(split.indices) == indices:
                print(f"\nSplit {indices}:")
                print(f"  In destination: YES")
                print(f"  In source: {split in source_splits}")
                taxa_list = list(split.taxa)
                print(
                    f"  Taxa: {taxa_list[:5]}..."
                    if len(taxa_list) > 5
                    else f"  Taxa: {taxa_list}"
                )
                break
        else:
            print(f"\nSplit {indices}: NOT FOUND in destination!")

    print("\n=== Running interpolation ===")
    result = process_tree_pair_interpolation(
        source_tree, destination_tree, pair_index=11
    )

    if result.trees:
        # Check each intermediate tree for the missing splits
        print(f"\nTotal intermediate trees: {len(result.trees)}")

        for indices in missing_indices:
            # Find the split in destination
            target_split = None
            for split in dest_splits:
                if tuple(split.indices) == indices:
                    target_split = split
                    break

            if target_split:
                print(f"\nTracking split {indices}:")
                present_in = []
                for i, tree in enumerate(result.trees):
                    tree_splits = tree.to_splits()
                    if target_split in tree_splits:
                        present_in.append(i)

                if present_in:
                    print(f"  Present in trees: {present_in}")
                    # Find where it disappears
                    for i in range(len(result.trees) - 1):
                        if i in present_in and (i + 1) not in present_in:
                            print(f"  DISAPPEARED after tree {i}")
                            # Check what pivot edge was being processed
                            if i < len(result.current_pivot_edge_tracking):
                                pivot = result.current_pivot_edge_tracking[i]
                                if pivot:
                                    print(
                                        f"  Pivot edge at tree {i}: {tuple(pivot.indices)[:5]}..."
                                    )
                else:
                    print(f"  NEVER present in any intermediate tree!")
                    print(f"  This split needs to be CREATED during interpolation")


if __name__ == "__main__":
    main()
