#!/usr/bin/env python3
"""Trace the interpolation error step by step."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.interpolation import build_sequential_lattice_interpolations
from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm
from brancharchitect.elements.partition import Partition


def trace_interpolation_error():
    """Trace what happens during interpolation."""
    # Define the two trees in Newick format
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    print("="*60)
    print("PARSING TREES SEPARATELY")
    print("="*60)
    
    # Parse trees separately
    tree1_sep = parse_newick(tree1_newick)
    tree2_sep = parse_newick(tree2_newick)
    
    print("\nTree 1 taxa order (separate parsing):")
    taxa1_sep = [leaf.name for leaf in tree1_sep.get_leaves()]
    print(f"  {[f'{i}:{name}' for i, name in enumerate(taxa1_sep)]}")
    
    print("\nTree 2 taxa order (separate parsing):")  
    taxa2_sep = [leaf.name for leaf in tree2_sep.get_leaves()]
    print(f"  {[f'{i}:{name}' for i, name in enumerate(taxa2_sep)]}")
    
    # Check specific partition
    partition_indices = (2, 3, 4, 5)
    print(f"\nPartition {partition_indices}:")
    print(f"  Tree 1: {[taxa1_sep[i] for i in partition_indices]}")
    print(f"  Tree 2: {[taxa2_sep[i] for i in partition_indices]}")
    
    print("\n" + "="*60)
    print("PARSING TREES TOGETHER")
    print("="*60)
    
    # Parse trees together
    tree_list = parse_newick(tree1_newick + tree2_newick)
    tree1_tog = tree_list[0]
    tree2_tog = tree_list[1]
    
    print("\nTree 1 taxa order (together parsing):")
    taxa1_tog = [leaf.name for leaf in tree1_tog.get_leaves()]
    print(f"  {[f'{i}:{name}' for i, name in enumerate(taxa1_tog)]}")
    
    print("\nTree 2 taxa order (together parsing):")
    taxa2_tog = [leaf.name for leaf in tree2_tog.get_leaves()]
    print(f"  {[f'{i}:{name}' for i, name in enumerate(taxa2_tog)]}")
    
    print(f"\nPartition {partition_indices}:")
    print(f"  Tree 1: {[taxa1_tog[i] for i in partition_indices]}")
    try:
        print(f"  Tree 2: {[taxa2_tog[i] for i in partition_indices]}")
    except IndexError as e:
        print(f"  Tree 2: ERROR - {e}")
    
    print("\n" + "="*60)
    print("LATTICE ALGORITHM COMPARISON")
    print("="*60)
    
    # Compare lattice results
    lattice_sep = iterate_lattice_algorithm(tree1_sep, tree2_sep)
    lattice_tog = iterate_lattice_algorithm(tree1_tog, tree2_tog)
    
    print(f"\nSeparate parsing: {len(lattice_sep)} s-edges")
    for edge in lattice_sep:
        print(f"  {edge.indices} -> {[taxa1_sep[i] for i in edge.indices]}")
    
    print(f"\nTogether parsing: {len(lattice_tog)} s-edges")
    for edge in lattice_tog:
        print(f"  {edge.indices} -> {[taxa1_tog[i] for i in edge.indices]}")
    
    print("\n" + "="*60)
    print("THE CORE ISSUE")
    print("="*60)
    
    print("\nThe problem occurs because:")
    print("1. When trees are parsed separately, they maintain consistent taxon indexing")
    print("2. When parsed together, the second tree gets different indices")
    print("3. The partition (2,3,4,5) exists in both trees with separate parsing")
    print("   but doesn't exist in tree 2 with together parsing")
    print("4. During interpolation, the algorithm tries to find nodes by partition indices")
    print("   that don't match between the consensus tree and reference tree")
    
    # Show the actual error location
    print("\n" + "="*60)
    print("FINDING THE PROBLEMATIC NODE")
    print("="*60)
    
    problem_partition = Partition((2, 3, 4, 5))
    
    # Check if this partition exists in each tree
    node1_sep = tree1_sep.find_node_by_split(problem_partition)
    node2_sep = tree2_sep.find_node_by_split(problem_partition)
    node1_tog = tree1_tog.find_node_by_split(problem_partition)
    node2_tog = tree2_tog.find_node_by_split(problem_partition)
    
    print(f"\nPartition {problem_partition.indices} exists in:")
    print(f"  Tree 1 (separate): {'YES' if node1_sep else 'NO'}")
    print(f"  Tree 2 (separate): {'YES' if node2_sep else 'NO'}")  
    print(f"  Tree 1 (together): {'YES' if node1_tog else 'NO'}")
    print(f"  Tree 2 (together): {'YES' if node2_tog else 'NO'}")
    
    if node2_sep:
        taxa = sorted([leaf.name for leaf in node2_sep.get_leaves()])
        print(f"\nIn Tree 2 (separate parsing), partition (2,3,4,5) contains: {taxa}")
        print(f"This is the node that should be found but isn't when parsing together!")


if __name__ == "__main__":
    trace_interpolation_error()