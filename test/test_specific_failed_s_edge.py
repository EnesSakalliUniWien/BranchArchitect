#!/usr/bin/env python3
"""Test to analyze the specific failed s-edge."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition


def test_specific_failed_s_edge():
    """Analyze the specific failed s-edge: (A1, A2, B1, C1, C2, D1, D2, X)."""
    # The exact trees
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse with correct encoding
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick, order=tree1._order)
    
    # Get the encoding to find indices for the failed taxa
    encoding = tree1.taxa_encoding
    failed_taxa = ['A1', 'A2', 'B1', 'C1', 'C2', 'D1', 'D2', 'X']
    
    print("ANALYZING FAILED S-EDGE")
    print("="*50)
    print(f"Failed taxa: {failed_taxa}")
    
    # Find the indices for these taxa
    failed_indices = tuple(sorted([encoding[taxon] for taxon in failed_taxa]))
    print(f"Failed indices: {failed_indices}")
    
    # Create the partition
    failed_partition = Partition(failed_indices)
    
    print(f"\nChecking if partition {failed_partition.indices} exists in both trees:")
    
    # Check Tree 1
    node1 = tree1.find_node_by_split(failed_partition)
    if node1:
        taxa1 = sorted([leaf.name for leaf in node1.get_leaves()])
        print(f"  Tree 1: ✓ FOUND -> {taxa1}")
        print(f"    Node structure: {node1.to_newick()}")
    else:
        print(f"  Tree 1: ✗ NOT FOUND")
        
        # Let's see what partitions Tree 1 does have that are close
        print(f"    Tree 1 splits containing some of these taxa:")
        splits1 = tree1.to_splits()
        for split in splits1:
            split_taxa = set([tree1.get_leaves()[i].name for i in split.indices])
            overlap = split_taxa.intersection(set(failed_taxa))
            if len(overlap) >= 4:  # Show splits with significant overlap
                print(f"      {split.indices} -> {sorted(split_taxa)} (overlap: {len(overlap)}/{len(failed_taxa)})")
    
    # Check Tree 2
    node2 = tree2.find_node_by_split(failed_partition)
    if node2:
        taxa2 = sorted([leaf.name for leaf in node2.get_leaves()])
        print(f"  Tree 2: ✓ FOUND -> {taxa2}")
        print(f"    Node structure: {node2.to_newick()}")
    else:
        print(f"  Tree 2: ✗ NOT FOUND")
        
        # Let's see what partitions Tree 2 does have that are close
        print(f"    Tree 2 splits containing some of these taxa:")
        splits2 = tree2.to_splits()
        for split in splits2:
            split_taxa = set([tree2.get_leaves()[i].name for i in split.indices])
            overlap = split_taxa.intersection(set(failed_taxa))
            if len(overlap) >= 4:  # Show splits with significant overlap
                print(f"      {split.indices} -> {sorted(split_taxa)} (overlap: {len(overlap)}/{len(failed_taxa)})")
    
    print(f"\n→ CONCLUSION:")
    if node1 and node2:
        print(f"  Both trees have this partition - this should NOT have failed!")
    elif node1:
        print(f"  Only Tree 1 has this partition - Tree 2 groups these taxa differently")
    elif node2:
        print(f"  Only Tree 2 has this partition - Tree 1 groups these taxa differently")
    else:
        print(f"  Neither tree has this exact partition - these taxa are never grouped together as a unit")
    
    # Show the full taxon-to-index mapping for reference
    print(f"\nTaxon-to-index mapping:")
    for taxon, index in sorted(encoding.items(), key=lambda x: x[1]):
        marker = "★" if taxon in failed_taxa else " "
        print(f"  {marker} {index:2d}: {taxon}")


if __name__ == "__main__":
    test_specific_failed_s_edge()