"""
Analyze split_matrix behavior across ALL test tree pairs.
"""

import json
import os
from pathlib import Path
from brancharchitect.parser.newick_parser import parse_newick as parse
from brancharchitect.jumping_taxa.lattice.solvers.pivot_edge_solver import LatticeSolver
from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    split_matrix,
)
from brancharchitect.jumping_taxa.lattice.matrices import build_conflict_matrix

# Find all test JSON files
test_dir = Path("test/colouring/trees")
json_files = list(test_dir.glob("**/*test*.json"))

split_occurred_cases = []
total_pivot_edges = 0
total_matrices = 0

for json_path in sorted(json_files):
    try:
        with open(json_path) as f:
            data = json.load(f)

        if "tree1" not in data or "tree2" not in data:
            continue

        tree1 = parse(data["tree1"])
        tree2 = parse(data["tree2"])
        tree2.taxa_encoding = tree1.taxa_encoding
        tree2.initialize_split_indices(tree1.taxa_encoding)

        solver = LatticeSolver(tree1, tree2)

        for pivot_edge in solver.pivot_edges:
            total_pivot_edges += 1
            candidate_matrix = build_conflict_matrix(pivot_edge)
            matrices = split_matrix(candidate_matrix)
            total_matrices += 1

            if len(matrices) > 1:
                split_occurred_cases.append(
                    {
                        "file": str(json_path),
                        "pivot": str(pivot_edge.pivot_split.bipartition()),
                        "matrix_rows": len(candidate_matrix),
                        "split_count": len(matrices),
                    }
                )
                print(
                    f"SPLIT! {json_path.name}: {len(matrices)} matrices from {len(candidate_matrix)} rows"
                )

    except Exception as e:
        print(f"Error processing {json_path.name}: {type(e).__name__}")

print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"Files processed: {len(json_files)}")
print(f"Total pivot edges: {total_pivot_edges}")
print(f"Total matrices: {total_matrices}")
print(f"Cases where split occurred (>1 matrices): {len(split_occurred_cases)}")

if split_occurred_cases:
    print(f"\nSplit cases:")
    for case in split_occurred_cases:
        print(
            f"  - {case['file']}: {case['split_count']} matrices from {case['matrix_rows']} rows"
        )
else:
    print(f"\n*** split_matrix NEVER returned >1 matrices across ALL test cases ***")
