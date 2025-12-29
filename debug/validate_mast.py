import argparse
import json
import logging
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import List, Set

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_trees_from_json(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)

    t1_newick = data["tree1"]
    t2_newick = data["tree2"]

    # Parse trees
    t1 = parse_newick(t1_newick)
    # Align taxa encoding
    t2 = parse_newick(t2_newick, list(t1.get_current_order()))

    return t1, t2, data.get("solutions", [])


def get_all_taxa(node: Node) -> Set[str]:
    return set(leaf.name for leaf in node.get_leaves())


def prune_taxa(tree: Node, taxa_to_remove: Set[str]) -> Node:
    """Return a deep copy of the tree with taxa_to_remove deleted."""
    # We must deep copy first because delete_taxa is in-place
    cloned = tree.deep_copy()

    # Map taxa names to indices for deletion
    indices_to_delete = []
    if cloned.taxa_encoding:
        for name in taxa_to_remove:
            if name in cloned.taxa_encoding:
                indices_to_delete.append(cloned.taxa_encoding[name])

    if indices_to_delete:
        cloned.delete_taxa(indices_to_delete)

    return cloned


def brute_force_mast(t1: Node, t2: Node):
    """
    Find Minimum Feedback Vertex Set (MAST) by brute-force deletion.
    Returns list of solutions (sets of taxa to remove).
    """
    taxa = sorted(list(get_all_taxa(t1)))
    n_taxa = len(taxa)

    logger.info(f"Total Taxa: {n_taxa}")
    logger.info(f"Taxa: {taxa}")

    solutions = []
    min_k = -1

    # Iterate k from 0 to N (number of taxa to remove)
    for k in range(n_taxa):
        logger.info(f"Checking removal of size {k}...")
        found_at_k = False

        for remove_set in combinations(taxa, k):
            remove_set = set(remove_set)

            # Prune trees
            t1_p = prune_taxa(t1, remove_set)
            t2_p = prune_taxa(t2, remove_set)

            # Check isomorphism
            if t1_p == t2_p:
                solutions.append(sorted(list(remove_set)))
                found_at_k = True

        if found_at_k:
            min_k = k
            break

    return solutions, min_k


def main():
    parser = argparse.ArgumentParser(description="Validate MAST solutions for a JSON test case.")
    parser.add_argument("json_file", help="Path to the JSON test file")
    args = parser.parse_args()

    path = Path(args.json_file)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    logger.info(f"Processing: {path.name}")
    t1, t2, expected_json = load_trees_from_json(str(path))

    actual_solutions, size = brute_force_mast(t1, t2)

    logger.info("\n=== BRUTE FORCE RESULTS ===")
    logger.info(f"Minimal Removal Size: {size}")
    logger.info("Solutions Found:")
    for sol in actual_solutions:
        logger.info(f"  {sol}")

    logger.info("\n=== JSON EXPECTED SOLUTIONS ===")
    for sol in expected_json:
        # Flatten structure [[X, Y], [Z]] -> [X, Y, Z] for comparison
        flat = sorted([item for sublist in sol for item in sublist])
        logger.info(f"  {flat} (Structure: {sol})")

    # Match check
    # Note: JSON solutions are fully partitioned. Brute force is just set of taxa.
    # We check if the set of names matches.

    flat_expected = []
    for sol in expected_json:
        flat_expected.append(sorted([item for sublist in sol for item in sublist]))

    matches = 0
    for sol in actual_solutions:
        if sol in flat_expected:
            matches += 1

    logger.info(f"\nMatched {matches}/{len(actual_solutions)} brute-force solutions with JSON expectations.")


if __name__ == "__main__":
    main()
