"""
Debug script to verify if both solutions for heiko_5_test_tree produce identical trees.
"""

import json
from brancharchitect.parser import parse_newick
from brancharchitect.tree import Node

# Read test data
with open("test/colouring/trees/heiko_5_test_tree/heiko_5_test_tree.json", "r") as f:
    test_data = json.load(f)

# Parse trees
tree1_str = test_data["tree1"]
tree2_str = test_data["tree2"]

print("=" * 80)
print("Testing heiko_5_test_tree.json")
print("=" * 80)

# Parse tree1 first to get encoding
tree1 = parse_newick(tree1_str)
if isinstance(tree1, list):
    tree1 = tree1[0]

print(f"\nTree 1 leaves: {sorted(tree1.taxa_encoding.keys())}")
print(f"Tree 1 newick: {tree1.to_newick()}")

# Parse tree2 with shared encoding
tree2 = parse_newick(
    tree2_str, order=list(tree1.taxa_encoding.keys()), encoding=tree1.taxa_encoding
)
if isinstance(tree2, list):
    tree2 = tree2[0]

print(f"Tree 2 leaves: {sorted(tree2.taxa_encoding.keys())}")
print(f"Tree 2 newick: {tree2.to_newick()}")

# Manually test expected solution 1: [('C1', 'C2'), ('E1', 'E2')]
print("\n" + "=" * 80)
print("TESTING EXPECTED SOLUTION 1: [('C1', 'C2'), ('E1', 'E2')]")
print("=" * 80)

tree1_copy1 = parse_newick(tree1_str)
if isinstance(tree1_copy1, list):
    tree1_copy1 = tree1_copy1[0]
tree2_copy1 = parse_newick(
    tree2_str,
    order=list(tree1_copy1.taxa_encoding.keys()),
    encoding=tree1_copy1.taxa_encoding,
)
if isinstance(tree2_copy1, list):
    tree2_copy1 = tree2_copy1[0]

# Delete C1, C2, E1, E2
taxa_to_delete = ["C1", "C2", "E1", "E2"]
print(f"\nDeleting: {taxa_to_delete}")

# Convert names to indices
indices_to_delete1 = [tree1_copy1.taxa_encoding[name] for name in taxa_to_delete]
indices_to_delete2 = [tree2_copy1.taxa_encoding[name] for name in taxa_to_delete]

tree1_after = tree1_copy1.delete_taxa(indices_to_delete1)
tree2_after = tree2_copy1.delete_taxa(indices_to_delete2)

print(
    f"Tree 1 after deletion leaves: {[leaf.name for leaf in tree1_after.get_leaves()]}"
)
print(
    f"Tree 2 after deletion leaves: {[leaf.name for leaf in tree2_after.get_leaves()]}"
)

newick1_after = tree1_after.to_newick()
newick2_after = tree2_after.to_newick()

print(f"\nTree 1 newick: {newick1_after}")
print(f"Tree 2 newick: {newick2_after}")
print(f"Trees identical: {newick1_after == newick2_after}")

# Manually test expected solution 2: [('C1', 'C2'), ('D1', 'D2')]
print("\n" + "=" * 80)
print("TESTING EXPECTED SOLUTION 2: [('C1', 'C2'), ('D1', 'D2')]")
print("=" * 80)

tree1_copy2 = parse_newick(tree1_str)
if isinstance(tree1_copy2, list):
    tree1_copy2 = tree1_copy2[0]
tree2_copy2 = parse_newick(
    tree2_str,
    order=list(tree1_copy2.taxa_encoding.keys()),
    encoding=tree1_copy2.taxa_encoding,
)
if isinstance(tree2_copy2, list):
    tree2_copy2 = tree2_copy2[0]

taxa_to_delete = ["C1", "C2", "D1", "D2"]
print(f"\nDeleting: {taxa_to_delete}")

# Convert names to indices
indices_to_delete1 = [tree1_copy2.taxa_encoding[name] for name in taxa_to_delete]
indices_to_delete2 = [tree2_copy2.taxa_encoding[name] for name in taxa_to_delete]

tree1_after2 = tree1_copy2.delete_taxa(indices_to_delete1)
tree2_after2 = tree2_copy2.delete_taxa(indices_to_delete2)

print(
    f"Tree 1 after deletion leaves: {[leaf.name for leaf in tree1_after2.get_leaves()]}"
)
print(
    f"Tree 2 after deletion leaves: {[leaf.name for leaf in tree2_after2.get_leaves()]}"
)

newick1_after2 = tree1_after2.to_newick()
newick2_after2 = tree2_after2.to_newick()

print(f"\nTree 1 newick: {newick1_after2}")
print(f"Tree 2 newick: {newick2_after2}")
print(f"Trees identical: {newick1_after2 == newick2_after2}")

# Compare parsimony
print("\n" + "=" * 80)
print("PARSIMONY COMPARISON:")
print("=" * 80)
print("Solution 1 [('C1', 'C2'), ('E1', 'E2')]: 4 taxa deleted")
print("Solution 2 [('C1', 'C2'), ('D1', 'D2')]: 4 taxa deleted")
print("\nBoth expected solutions have EQUAL parsimony (4 taxa)!")
print(f"Solution 1 makes trees identical: {newick1_after == newick2_after}")
print(f"Solution 2 makes trees identical: {newick1_after2 == newick2_after2}")
