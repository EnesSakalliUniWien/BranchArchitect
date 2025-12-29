#!/usr/bin/env python
"""
Debug the tree structure to understand why reorder_taxa can't move Ostrich.
"""

from brancharchitect.io import read_newick


def print_newick(tree, label):
    """Print the newick string of a tree."""
    print(f"\n{label} Newick:")
    print(tree.to_newick(lengths=False))


def print_tree_structure(node, indent=0):
    """Print tree structure with split indices."""
    prefix = "  " * indent
    if node.children:
        taxa = (
            sorted(node.split_indices.taxa)
            if hasattr(node.split_indices, "taxa")
            else str(node.split_indices)
        )
        print(
            f"{prefix}Internal: {taxa[:5]}..."
            if len(str(taxa)) > 50
            else f"{prefix}Internal: {taxa}"
        )
        for child in node.children:
            print_tree_structure(child, indent + 1)
    else:
        print(f"{prefix}Leaf: {node.name}")


def find_ostrich_path(node, path=None):
    """Find the path from root to Ostrich."""
    if path is None:
        path = []

    path.append(node)

    if not node.children:
        if node.name == "Ostrich":
            return path
        return None

    for child in node.children:
        result = find_ostrich_path(child, path.copy())
        if result:
            return result

    return None


def main():
    trees = read_newick("small_example copy 3.tree", treat_zero_as_epsilon=True)

    source = trees[3]
    dest = trees[4]

    # Print newick strings first
    print_newick(source, "SOURCE TREE (Tree 3)")
    print_newick(dest, "DESTINATION TREE (Tree 4)")

    print("\n" + "=" * 80)
    print("SOURCE TREE (Tree 3) STRUCTURE")
    print("=" * 80)
    print_tree_structure(source)

    print("\n" + "=" * 80)
    print("DESTINATION TREE (Tree 4) STRUCTURE")
    print("=" * 80)
    print_tree_structure(dest)

    print("\n" + "=" * 80)
    print("PATH TO OSTRICH IN SOURCE")
    print("=" * 80)
    path = find_ostrich_path(source)
    if path:
        for i, node in enumerate(path):
            if node.children:
                taxa = (
                    sorted(node.split_indices.taxa)
                    if hasattr(node.split_indices, "taxa")
                    else str(node.split_indices)
                )
                print(f"Level {i}: Internal node with {len(node.children)} children")
                print(
                    f"         Taxa: {taxa[:10]}..."
                    if len(str(taxa)) > 80
                    else f"         Taxa: {taxa}"
                )
            else:
                print(f"Level {i}: Leaf '{node.name}'")

    print("\n" + "=" * 80)
    print("PATH TO OSTRICH IN DESTINATION")
    print("=" * 80)
    path = find_ostrich_path(dest)
    if path:
        for i, node in enumerate(path):
            if node.children:
                taxa = (
                    sorted(node.split_indices.taxa)
                    if hasattr(node.split_indices, "taxa")
                    else str(node.split_indices)
                )
                print(f"Level {i}: Internal node with {len(node.children)} children")
                print(
                    f"         Taxa: {taxa[:10]}..."
                    if len(str(taxa)) > 80
                    else f"         Taxa: {taxa}"
                )
            else:
                print(f"Level {i}: Leaf '{node.name}'")

    # Check the siblings of Ostrich in both trees
    print("\n" + "=" * 80)
    print("OSTRICH'S SIBLINGS IN SOURCE")
    print("=" * 80)
    path = find_ostrich_path(source)
    if path and len(path) >= 2:
        parent = path[-2]
        print(f"Parent has {len(parent.children)} children:")
        for child in parent.children:
            if child.children:
                taxa = sorted(child.split_indices.taxa)
                print(f"  Internal: {taxa}")
            else:
                print(f"  Leaf: {child.name}")

    print("\n" + "=" * 80)
    print("OSTRICH'S SIBLINGS IN DESTINATION")
    print("=" * 80)
    path = find_ostrich_path(dest)
    if path and len(path) >= 2:
        parent = path[-2]
        print(f"Parent has {len(parent.children)} children:")
        for child in parent.children:
            if child.children:
                taxa = sorted(child.split_indices.taxa)
                print(f"  Internal: {taxa}")
            else:
                print(f"  Leaf: {child.name}")

    # The key insight: in source, Ostrich is grouped with (GreatRhea, LesserRhea)
    # In destination, Ostrich is grouped with the Moa/Tinamou clade
    # reorder_taxa can only reorder children within a node, not move leaves across clades

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
The issue is that reorder_taxa() can only reorder children within a node.
It cannot move a leaf (Ostrich) from one clade to another.

In the SOURCE tree:
- Ostrich is a sibling of (GreatRhea, LesserRhea) under a common parent

In the DESTINATION tree:
- Ostrich is in a different position in the tree topology

The reorder_tree_toward_destination function computes the correct target order,
but reorder_taxa() cannot achieve it because it would require changing the
tree topology, not just reordering children.

This is a fundamental limitation: reorder_taxa() preserves tree topology
while only changing the visual order of children at each node.
""")


if __name__ == "__main__":
    main()
