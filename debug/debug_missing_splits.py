#!/usr/bin/env python3
"""Debug script to trace where splits are being lost during interpolation."""

import logging
from pathlib import Path
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)

# Suppress most logging
logging.basicConfig(level=logging.WARNING)


def load_trees(filepath: str) -> list[Node]:
    """Load trees from a Newick file."""
    with open(filepath, "r") as f:
        content = f.read()

    result = parse_newick(content, force_list=True)
    if isinstance(result, list):
        return result
    return [result]


def find_missing_splits(final_tree: Node, destination_tree: Node) -> tuple[set, set]:
    """Find splits that are missing or extra in the final tree."""
    final_splits = final_tree.to_splits()
    dest_splits = destination_tree.to_splits()

    missing = dest_splits - final_splits
    extra = final_splits - dest_splits

    return missing, extra


def main():
    # Load the test dataset
    filepath = "datasets/all_trees copy 5.nwk"
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return

    trees = load_trees(filepath)
    print(
        f"Loaded {len(trees)} trees with {len(list(trees[0].get_current_order()))} leaves each"
    )

    # Test all consecutive pairs
    failures = []
    for i in range(len(trees) - 1):
        source_tree = trees[i]
        destination_tree = trees[i + 1]

        # Ensure same encoding
        destination_tree.initialize_split_indices(source_tree.taxa_encoding)

        result = process_tree_pair_interpolation(
            source_tree, destination_tree, pair_index=i
        )

        if result.trees:
            final_tree = result.trees[-1]
            missing, extra = find_missing_splits(final_tree, destination_tree)

            if missing or extra:
                failures.append(
                    {
                        "pair": i,
                        "missing": missing,
                        "extra": extra,
                        "trees": result.trees,
                    }
                )
                print(f"Pair {i}: FAILED - {len(missing)} missing, {len(extra)} extra")
            else:
                print(f"Pair {i}: OK")
        else:
            print(f"Pair {i}: No trees generated!")

    print(f"\n{'=' * 60}")
    print(f"Total failures: {len(failures)}")

    for failure in failures:
        print(f"\n--- Pair {failure['pair']} ---")
        if failure["missing"]:
            print(f"Missing splits ({len(failure['missing'])}):")
            for split in sorted(failure["missing"], key=lambda s: len(s.indices)):
                print(f"  {tuple(split.indices)}")
                # Check if this split was ever in any intermediate tree
                found_in = []
                for j, tree in enumerate(failure["trees"]):
                    if split in tree.to_splits():
                        found_in.append(j)
                if found_in:
                    print(f"    -> Was present in trees: {found_in[:10]}...")
                else:
                    print("    -> NEVER present in any intermediate tree!")
        if failure["extra"]:
            print(f"Extra splits ({len(failure['extra'])}):")
            for split in sorted(failure["extra"], key=lambda s: len(s.indices)):
                print(f"  {tuple(split.indices)}")


if __name__ == "__main__":
    main()
