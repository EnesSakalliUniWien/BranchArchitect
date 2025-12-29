import json
import logging
import sys
import os
from typing import Set, Tuple, List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

# Set up raw logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("failure_analyzer")

def format_split(split, encoding: Dict[str, int]) -> str:
    """Format a split (partition) as string of taxon names."""
    indices = split.indices
    names = sorted([k for k, v in encoding.items() if v in indices])
    return f"{{{', '.join(names)}}}"

def analyze_failure(json_path: str):
    logger.info(f"\nAnalyzing: {json_path}")
    logger.info("=" * 60)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return

    try:
        t1_orig = parse_newick(data["tree1"])
        if isinstance(t1_orig, list): t1_orig = t1_orig[0]

        t2_orig = parse_newick(data["tree2"], encoding=t1_orig.taxa_encoding)
        if isinstance(t2_orig, list): t2_orig = t2_orig[0]
    except Exception as e:
        logger.error(f"Failed to parse trees: {e}")
        return

    encoding = t1_orig.taxa_encoding

    for i, solution_set in enumerate(data.get("solutions", [])):
        candidate_taxa = []
        for partition in solution_set:
            candidate_taxa.extend(partition)

        logger.info(f"\nSolution #{i+1}: Removing {candidate_taxa}")

        # Deep copy
        t1 = t1_orig.deep_copy()
        t2 = t2_orig.deep_copy()

        # Prune
        indices_to_delete = [encoding[t] for t in candidate_taxa if t in encoding]
        if indices_to_delete:
            t1.delete_taxa(indices_to_delete)
            t2.delete_taxa(indices_to_delete)

        # Compare Topological Splits
        splits1 = t1.to_splits(with_leaves=True)
        splits2 = t2.to_splits(with_leaves=True)

        if splits1 == splits2:
            logger.info("  [OK] Pruned trees are isomeric.")
        else:
            logger.info("  [FAIL] Pruned trees differ.")

            # Find symmetric difference
            diff1 = splits1 - splits2 # In 1 but not 2
            diff2 = splits2 - splits1 # In 2 but not 1

            if diff1:
                logger.info("    Splits in Tree 1 BUT NOT Tree 2:")
                for s in sorted(diff1, key=lambda x: len(x.indices)):
                    logger.info(f"      {format_split(s, encoding)}")

            if diff2:
                logger.info("    Splits in Tree 2 BUT NOT Tree 1:")
                for s in sorted(diff2, key=lambda x: len(x.indices)):
                    logger.info(f"      {format_split(s, encoding)}")

            logger.info(f"    Newick 1: {t1.to_newick()}")
            logger.info(f"    Newick 2: {t2.to_newick()}")

if __name__ == "__main__":
    files_to_analyze = [
        "test/colouring/trees/test_tree_moving_updwards/test_tree_moving_updwards.json",
        "test/colouring/trees/basic_1_taxon_partial/basic_1_taxon_partial.json",
        "test/colouring/trees/test_tree_moving_updwards/reverse_test_tree_moving_updwards.json",
         "test/colouring/trees/simon_test_tree_7/reverse_simon_test_tree_7.json"
    ]

    for f in files_to_analyze:
        analyze_failure(f)
