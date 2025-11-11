"""
Comprehensive analysis of both failing tests.
Checking if trees are topologically identical even if newick strings differ.
"""

import json
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa


def analyze_topology(test_file, test_name):
    print("=" * 80)
    print(f"ANALYSIS: {test_name}")
    print("=" * 80)

    with open(test_file, "r") as f:
        data = json.load(f)

    tree1_str = data["tree1"]
    tree2_str = data["tree2"]

    print(f"\nOriginal trees:")
    print(f"Tree 1: {tree1_str}")
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

    print(f"\nTaxa: {sorted(tree1.taxa_encoding.keys())}")

    # Run algorithm
    result = call_jumping_taxa(tree1, tree2)

    # Convert to names
    idx_to_name = {idx: name for name, idx in tree1.taxa_encoding.items()}
    solution_names = []
    for partition_tuple in result:
        partition_names = tuple(sorted([idx_to_name[idx] for idx in partition_tuple]))
        solution_names.append(partition_names)
    solution_names = sorted(solution_names)

    print(f"\nAlgorithm solution: {solution_names}")
    print(f"Total jumping taxa: {sum(len(p) for p in solution_names)}")

    # Delete and check
    indices_to_delete = [idx for partition in result for idx in partition]
    deleted_names = sorted([idx_to_name[idx] for idx in indices_to_delete])

    tree1_after = tree1.delete_taxa(indices_to_delete)
    tree2_after = tree2.delete_taxa(indices_to_delete)

    remaining1 = sorted([leaf.name for leaf in tree1_after.get_leaves()])
    remaining2 = sorted([leaf.name for leaf in tree2_after.get_leaves()])

    print(f"\nDeleted: {deleted_names}")
    print(f"Remaining: {remaining1}")

    newick1 = tree1_after.to_newick()
    newick2 = tree2_after.to_newick()

    print(f"\nTree 1: {newick1}")
    print(f"Tree 2: {newick2}")
    print(f"Newick strings identical: {newick1 == newick2}")

    # Check topological equivalence using splits
    print("\n--- Checking topological equivalence ---")

    # Use built-in equality check which compares split_indices
    topologically_identical = tree1_after == tree2_after
    print(f"Using tree equality (split_indices): {topologically_identical}")

    # Also manually check splits for detailed comparison
    splits1 = set()
    for node in tree1_after.traverse():
        if not node.is_leaf():
            leaf_names = frozenset([leaf.name for leaf in node.get_leaves()])
            if len(leaf_names) > 1 and len(leaf_names) < len(remaining1):
                splits1.add(leaf_names)

    splits2 = set()
    for node in tree2_after.traverse():
        if not node.is_leaf():
            leaf_names = frozenset([leaf.name for leaf in node.get_leaves()])
            if len(leaf_names) > 1 and len(leaf_names) < len(remaining2):
                splits2.add(leaf_names)

    print(f"\nTree 1 splits ({len(splits1)}):")
    for split in sorted([sorted(s) for s in splits1]):
        print(f"  {split}")

    print(f"\nTree 2 splits ({len(splits2)}):")
    for split in sorted([sorted(s) for s in splits2]):
        print(f"  {split}")

    # Check if splits are identical
    splits_identical = splits1 == splits2
    print(f"\n{'✓' if splits_identical else '✗'} Splits identical: {splits_identical}")

    if not topologically_identical:
        print("\nMissing in Tree 1:")
        for split in splits2 - splits1:
            print(f"  {sorted(split)}")
        print("\nMissing in Tree 2:")
        for split in splits1 - splits2:
            print(f"  {sorted(split)}")

    return {
        "solution": solution_names,
        "newick_identical": newick1 == newick2,
        "topologically_identical": topologically_identical,
        "remaining_taxa": len(remaining1),
    }


# Analyze both tests
result1 = analyze_topology(
    "test/colouring/trees/test-070825/test_070825.json", "test_070825.json"
)
print("\n\n")
result2 = analyze_topology(
    "test/colouring/trees/test-13-08-24/test_130825.json", "test_130825.json"
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\ntest_070825.json:")
print(f"  Solution: {result1['solution']}")
print(f"  Newick identical: {result1['newick_identical']}")
print(f"  Topologically identical: {result1['topologically_identical']}")
print(f"  Remaining taxa: {result1['remaining_taxa']}")

print(f"\ntest_130825.json:")
print(f"  Solution: {result2['solution']}")
print(f"  Newick identical: {result2['newick_identical']}")
print(f"  Topologically identical: {result2['topologically_identical']}")
print(f"  Remaining taxa: {result2['remaining_taxa']}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if result1["topologically_identical"] and result2["topologically_identical"]:
    print("✓ Both solutions produce topologically identical trees!")
    print("✓ Algorithm is CORRECT - test expectations are WRONG")
    print("✗ Tree comparison is using newick string equality instead of topology")
else:
    print("✗ Trees are NOT topologically identical")
    print("✗ Algorithm has a bug - found incomplete solutions")
