"""
Debug script to analyze the two failing tests and determine correct solutions.
"""

import json
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa


def analyze_test(test_file):
    print("=" * 80)
    print(f"Analyzing: {test_file}")
    print("=" * 80)

    with open(test_file, "r") as f:
        data = json.load(f)

    tree1_str = data["tree1"]
    tree2_str = data["tree2"]

    print(f"\nTree 1: {tree1_str}")
    print(f"Tree 2: {tree2_str}")

    # Parse trees
    tree1 = parse_newick(tree1_str)
    if isinstance(tree1, list):
        tree1 = tree1[0]

    tree2 = parse_newick(
        tree2_str, order=list(tree1.taxa_encoding.keys()), encoding=tree1.taxa_encoding
    )
    if isinstance(tree2, list):
        tree2 = tree2[0]

    print(f"\nTree 1 taxa: {sorted(tree1.taxa_encoding.keys())}")
    print(f"Tree 2 taxa: {sorted(tree2.taxa_encoding.keys())}")

    # Run jumping taxa algorithm
    result = call_jumping_taxa(tree1, tree2)

    # Convert indices to names
    idx_to_name = {idx: name for name, idx in tree1.taxa_encoding.items()}
    solution_names = []
    for partition_tuple in result:
        partition_names = tuple(sorted([idx_to_name[idx] for idx in partition_tuple]))
        solution_names.append(partition_names)

    solution_names = sorted(solution_names)

    print(f"\nAlgorithm found solution: {solution_names}")
    print(f"Total jumping taxa: {sum(len(p) for p in solution_names)}")
    print(f"\nExpected solutions in file: {data['solutions']}")
    print(f"Comment: {data['comment']}")

    # Verify the solution makes trees identical
    print("\n--- Verifying solution correctness ---")
    indices_to_delete = [idx for partition in result for idx in partition]
    tree1_after = tree1.delete_taxa(indices_to_delete)
    tree2_after = tree2.delete_taxa(indices_to_delete)

    newick1 = tree1_after.to_newick()
    newick2 = tree2_after.to_newick()

    print(f"Tree 1 after deletion: {newick1}")
    print(f"Tree 2 after deletion: {newick2}")
    print(f"Trees identical: {newick1 == newick2}")

    return solution_names


# Analyze both tests
solution1 = analyze_test("test/colouring/trees/test-070825/test_070825.json")
print("\n\n")
solution2 = analyze_test("test/colouring/trees/test-13-08-24/test_130825.json")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\ntest_070825.json should have solution: {solution1}")
print(f"test_130825.json should have solution: {solution2}")
