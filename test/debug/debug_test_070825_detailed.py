"""
Detailed analysis of test_070825.json failure.
This test shows trees are NOT identical after deletion - investigating why.
"""

import json
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa

print("=" * 80)
print("DETAILED ANALYSIS: test_070825.json")
print("=" * 80)

with open("test/colouring/trees/test-070825/test_070825.json", "r") as f:
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

# Get initial newicks
print("\n" + "=" * 80)
print("INITIAL TREES (formatted)")
print("=" * 80)
print(f"Tree 1: {tree1.to_newick()}")
print(f"Tree 2: {tree2.to_newick()}")

# Run jumping taxa algorithm
print("\n" + "=" * 80)
print("ALGORITHM EXECUTION")
print("=" * 80)
result = call_jumping_taxa(tree1, tree2)

# Convert indices to names
idx_to_name = {idx: name for name, idx in tree1.taxa_encoding.items()}
solution_names = []
for partition_tuple in result:
    partition_names = tuple(sorted([idx_to_name[idx] for idx in partition_tuple]))
    solution_names.append(partition_names)

solution_names = sorted(solution_names)

print(f"\nAlgorithm found: {solution_names}")
print(f"Total jumping taxa: {sum(len(p) for p in solution_names)}")

# Delete and verify
print("\n" + "=" * 80)
print("AFTER DELETION")
print("=" * 80)
indices_to_delete = [idx for partition in result for idx in partition]
deleted_names = [idx_to_name[idx] for idx in indices_to_delete]
print(f"Deleting indices: {indices_to_delete}")
print(f"Deleting taxa names: {deleted_names}")

tree1_after = tree1.delete_taxa(indices_to_delete)
tree2_after = tree2.delete_taxa(indices_to_delete)

print(f"\nTree 1 remaining leaves: {[leaf.name for leaf in tree1_after.get_leaves()]}")
print(f"Tree 2 remaining leaves: {[leaf.name for leaf in tree2_after.get_leaves()]}")

newick1 = tree1_after.to_newick()
newick2 = tree2_after.to_newick()

print(f"\nTree 1 newick: {newick1}")
print(f"Tree 2 newick: {newick2}")

# Check if identical
identical = newick1 == newick2
print(f"\n{'✓' if identical else '✗'} Trees identical: {identical}")

if not identical:
    print("\n" + "=" * 80)
    print("PROBLEM ANALYSIS")
    print("=" * 80)
    print("Trees are NOT identical after deletion!")
    print("\nComparing structures:")

    # Parse both newicks to compare
    leaves1 = [leaf.name for leaf in tree1_after.get_leaves()]
    leaves2 = [leaf.name for leaf in tree2_after.get_leaves()]

    print(f"Tree 1 leaves in order: {leaves1}")
    print(f"Tree 2 leaves in order: {leaves2}")

    # Get all splits
    from brancharchitect.elements.split import Split

    print("\nTree 1 splits:")
    for split in tree1_after.get_splits():
        print(f"  {split}")

    print("\nTree 2 splits:")
    for split in tree2_after.get_splits():
        print(f"  {split}")

    print(
        "\n⚠️  CONCLUSION: Algorithm found incomplete solution or trees have remaining conflicts"
    )
