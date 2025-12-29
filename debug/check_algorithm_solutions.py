import json
import logging
import sys
import os
import sys
import os
from unittest.mock import MagicMock

# MOCK TABULATE to avoid dependency error
mock_tabulate = MagicMock()
mock_tabulate.tabulate = lambda data, headers=None, tablefmt=None: str(data)
sys.modules["tabulate"] = mock_tabulate

from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.verification import verify_jumping_taxa_solution
from brancharchitect.jumping_taxa.api import call_jumping_taxa

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("algo_checker")

def check_algorithm_on_file(json_path: str):
    logger.info(f"\n==================================================")
    logger.info(f"Running Algorithm on: {json_path}")
    logger.info(f"==================================================")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return

    try:
        t1 = parse_newick(data["tree1"])
        if isinstance(t1, list): t1 = t1[0]

        t2 = parse_newick(data["tree2"], encoding=t1.taxa_encoding)
        if isinstance(t2, list): t2 = t2[0]
    except Exception as e:
        logger.error(f"Failed to parse trees: {e}")
        return

    # Run Algorithm
    logger.info("Running call_jumping_taxa(algorithm='lattice')...")
    try:
        # call_jumping_taxa returns list[tuple[int]]
        # We need to rely on t1.taxa_encoding to decode these
        solutions_indices = call_jumping_taxa(t1, t2, algorithm="lattice")
    except Exception as e:
        logger.error(f"Algorithm failed with crashing error: {e}")
        return

    logger.info(f"Algorithm returned {len(solutions_indices)} solutions.")

    reverse_encoding = {v: k for k, v in t1.taxa_encoding.items()}

    # Check each solution
    for i, sol_indices in enumerate(solutions_indices):
        # Decode
        candidate_taxa = []
        for idx in sol_indices:
            name = reverse_encoding.get(idx)
            if name:
                candidate_taxa.append(name)
            else:
                candidate_taxa.append(f"UNKNOWN_IDX_{idx}")

        logger.info(f"\n  Algo Solution #{i+1}: {candidate_taxa}")

        # Verify
        report = verify_jumping_taxa_solution(t1, t2, candidate_taxa)

        if report["success"]:
            logger.info("    [PASS] Verification successful.")
        else:
            logger.info("    [FAIL] Verification failed.")
            for err in report["errors"]:
                logger.info(f"      - {err}")

    # Also compare with JSON expected solutions (just names)
    expected_sets = []
    for sol_set in data.get("solutions", []):
         flat = []
         for p in sol_set: flat.extend(p)
         expected_sets.append(set(flat))

    # Check if found solutions match any expected
    logger.info("\n  Comparison with JSON expected solutions:")
    for i, sol_indices in enumerate(solutions_indices):
        candidate_set = set()
        for idx in sol_indices:
             name = reverse_encoding.get(idx)
             if name: candidate_set.add(name)

        match_found = False
        for expected in expected_sets:
            if candidate_set == expected:
                match_found = True
                break

        if match_found:
             logger.info(f"    Algo Solution #{i+1} matches an expected solution in JSON.")
        else:
             logger.info(f"    Algo Solution #{i+1} does NOT match any expected solution in JSON.")


if __name__ == "__main__":
    files_to_check = [
        "test/colouring/trees/test_tree_moving_updwards/test_tree_moving_updwards.json",
        "test/colouring/trees/basic_1_taxon_partial/basic_1_taxon_partial.json",
        "test/colouring/trees/simon_test_tree_7/reverse_simon_test_tree_7.json"
    ]

    for f in files_to_check:
        check_algorithm_on_file(f)
