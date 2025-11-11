#!/usr/bin/env python3
"""Compute actual solutions for test cases and update JSON files."""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from brancharchitect.jumping_taxa import call_jumping_taxa
from brancharchitect.parser.newick_parser import parse_newick


def compute_solution(tree1_str: str, tree2_str: str):
    """Compute the jumping taxa solution for two trees."""
    t1 = parse_newick(tree1_str)
    t2 = parse_newick(tree2_str, list(t1.get_current_order()))

    result = call_jumping_taxa(t1, t2, algorithm="lattice")

    # Convert indices to names
    reverse_encoding = {v: k for k, v in t1.taxa_encoding.items()}
    solutions = []
    for solution_set in result:
        taxa_names = sorted(
            [reverse_encoding[idx] for part in solution_set for idx in part]
        )
        solutions.append([[name] for name in taxa_names])

    return solutions


def update_test_file(filepath: Path):
    """Update a test JSON file with computed solutions."""
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"Processing: {filepath.name}")
    print(f"{'=' * 60}")

    solutions = compute_solution(data["tree1"], data["tree2"])

    print(f"Tree 1: {data['tree1']}")
    print(f"Tree 2: {data['tree2']}")
    print(f"\nComputed solutions: {solutions}")
    print(f"Original solutions: {data.get('solutions', 'NONE')}")

    data["solutions"] = solutions

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✓ Updated {filepath.name}")


if __name__ == "__main__":
    test_dir = project_root / "test" / "colouring" / "trees" / "taxon_crossing_root"

    json_files = sorted(test_dir.glob("*.json"))

    for json_file in json_files:
        try:
            update_test_file(json_file)
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
            import traceback

            traceback.print_exc()
