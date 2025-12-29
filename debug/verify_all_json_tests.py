import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.verification import verify_jumping_taxa_solution


def load_and_verify_json(file_path: str) -> None:
    print(f"\n==================================================")
    print(f"Verifying file: {file_path}")
    print(f"==================================================")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        return

    if "tree1" not in data or "tree2" not in data:
        print("SKIPPING: File does not contain 'tree1' and 'tree2' fields.")
        return

    try:
        t1 = parse_newick(data["tree1"])
        # Ensure encoding consistency for creating tree2
        if isinstance(t1, list):
             t1 = t1[0]

        t2 = parse_newick(data["tree2"], encoding=t1.taxa_encoding)
        if isinstance(t2, list):
             t2 = t2[0]

    except Exception as e:
        print(f"ERROR: Failed to parse trees: {e}")
        return

    sentences = data.get("solutions", [])
    if not sentences:
        print("WARNING: No 'solutions' found in JSON.")
        return

    all_passed = True

    for i, solution_set in enumerate(sentences):
        # Flatten the solution set: solution_set is List[List[str]] (list of partitions)
        # We need a flat list of taxa strings for the verifier
        candidate_taxa = []
        for partition in solution_set:
            candidate_taxa.extend(partition)

        print(f"\n  Checking Solution #{i+1}: {candidate_taxa}")

        result = verify_jumping_taxa_solution(t1, t2, candidate_taxa)

        if result["success"]:
            print(f"    [PASS] Verification successful.")
        else:
            all_passed = False
            print(f"    [FAIL] Verification failed.")
            for err in result["errors"]:
                print(f"      - Error: {err}")
            for warn in result["warnings"]:
                print(f"      - Warning: {warn}")
            print(f"      - Metrics before: {result['metrics_before']}")
            print(f"      - Metrics after:  {result['metrics_after']}")

    if all_passed:
        print(f"\nResult: ALL SOLUTIONS VERIFIED for {os.path.basename(file_path)}")
    else:
        print(f"\nResult: SOME SOLUTIONS FAILED for {os.path.basename(file_path)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Verify specific provided files
        files = sys.argv[1:]
    else:
        # Default: verify all .json files in test/colouring/trees recursively
        files = []
        start_dir = "test/colouring/trees"
        for root, dirs, filenames in os.walk(start_dir):
            for filename in filenames:
                if filename.endswith(".json"):
                    files.append(os.path.join(root, filename))

    print(f"Found {len(files)} files to verify.")
    for f in sorted(files):
        load_and_verify_json(f)
