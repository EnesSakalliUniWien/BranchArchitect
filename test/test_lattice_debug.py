#!/usr/bin/env python3
"""Debug test to understand lattice edges."""

from typing import List
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import (
    iterate_lattice_algorithm,
)
from brancharchitect.elements.partition import Partition


def test_lattice_edges():
    """Test to understand what lattice edges are."""
    # Define the two trees in Newick format
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"

    # Parse the trees
    tree_list: Node | List[Node] = parse_newick(tree1_newick + tree2_newick)

    # Extract individual trees from the list
    if isinstance(tree_list, list) and len(tree_list) >= 2:
        tree1 = tree_list[0]
        tree2 = tree_list[1]
    else:
        raise ValueError("Expected to parse 2 trees from the Newick strings")

    print("Tree 1 (Target):")
    print(tree1.to_newick())
    print("\nTree 2 (Reference):")
    print(tree2.to_newick())

    # Get splits from both trees
    splits1: PartitionSet[Partition] = tree1.to_splits()
    splits2: PartitionSet[Partition] = tree2.to_splits()

    # Get leaf names
    leaf_names1: List[str] = [leaf.name for leaf in tree1.get_leaves()]
    leaf_names2: List[str] = [leaf.name for leaf in tree2.get_leaves()]

    print("\n" + "=" * 60)
    print("TREE SPLITS (PARTITIONS)")
    print("=" * 60)

    print("\nTree 1 splits:")
    for split in splits1:
        print(f"  {split.indices} -> Taxa: {[leaf_names1[i] for i in split.indices]}")

    print("\nTree 2 splits:")
    for split in splits2:
        print(f"  {split.indices} -> Taxa: {[leaf_names2[i] for i in split.indices]}")

    # Run the lattice algorithm
    print("\n" + "=" * 60)
    print("LATTICE ALGORITHM RESULTS")
    print("=" * 60)

    lattice_edge_solutions = iterate_lattice_algorithm(tree1, tree2)

    print(f"\nFound {len(lattice_edge_solutions)} lattice edges (s-edges):")

    for i, (edge, solutions) in enumerate(lattice_edge_solutions.items()):
        taxa_names = [leaf_names1[idx] for idx in edge.indices]
        print(f"\n{i + 1}. Lattice Edge: {edge.indices}")
        print(f"   Taxa: {taxa_names}")
        print(f"   Number of solution sets: {len(solutions)}")

        for j, solution_set in enumerate(solutions):
            print(f"   Solution set {j + 1}:")
            for partition in solution_set:
                part_taxa = [leaf_names1[idx] for idx in partition.indices]
                print(f"     - Partition {partition.indices} -> Taxa: {part_taxa}")

    # Check which splits are unique to each tree
    print("\n" + "=" * 60)
    print("SPLIT DIFFERENCES")
    print("=" * 60)

    unique_to_tree1: PartitionSet[Partition] = splits1 - splits2
    unique_to_tree2: PartitionSet[Partition] = splits2 - splits1
    common_splits: PartitionSet[Partition] = splits1 & splits2

    print(f"\nCommon splits: {len(common_splits)}")
    for split in common_splits:
        taxa = [leaf_names1[i] for i in split.indices]
        print(f"  {split.indices} -> {taxa}")

    print(f"\nUnique to Tree 1: {len(unique_to_tree1)}")
    for split in unique_to_tree1:
        taxa = [leaf_names1[i] for i in split.indices]
        print(f"  {split.indices} -> {taxa}")

    print(f"\nUnique to Tree 2: {len(unique_to_tree2)}")
    for split in unique_to_tree2:
        taxa = [leaf_names2[i] for i in split.indices]
        print(f"  {split.indices} -> {taxa}")

    # Now let's trace the specific edge that's causing problems
    print("\n" + "=" * 60)
    print("PROBLEMATIC EDGE ANALYSIS")
    print("=" * 60)

    problem_edge = Partition((2, 3, 4, 5))  # B1, C1, C2, X
    print(f"\nAnalyzing edge {problem_edge.indices}:")

    # Find this split in both trees
    node1 = tree1.find_node_by_split(problem_edge)
    node2 = tree2.find_node_by_split(problem_edge)

    if node1:
        taxa1 = sorted([leaf.name for leaf in node1.get_leaves()])
        print(f"  In Tree 1: Found node with taxa {taxa1}")
        print(f"  Node structure: {node1.to_newick()}")
    else:
        print("  In Tree 1: NOT FOUND")

    if node2:
        taxa2 = sorted([leaf.name for leaf in node2.get_leaves()])
        print(f"  In Tree 2: Found node with taxa {taxa2}")
        print(f"  Node structure: {node2.to_newick()}")
    else:
        print("  In Tree 2: NOT FOUND")


if __name__ == "__main__":
    test_lattice_edges()
