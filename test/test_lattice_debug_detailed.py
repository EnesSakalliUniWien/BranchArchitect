#!/usr/bin/env python3
"""Detailed debugging of lattice construction to understand the s-edge issue."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_construction import construct_sub_lattices
from brancharchitect.elements.partition import Partition


def test_lattice_debug_detailed():
    """Debug lattice construction in detail."""
    # The exact trees
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse with correct encoding
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick, order=tree1._order)
    
    print("DETAILED LATTICE CONSTRUCTION ANALYSIS")
    print("="*60)
    
    # Get splits directly
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()
    
    print(f"Tree 1 splits: {len(splits1)}")
    for split in sorted(splits1, key=lambda x: x.indices):
        taxa = [tree1.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    print(f"\nTree 2 splits: {len(splits2)}")
    for split in sorted(splits2, key=lambda x: x.indices):
        taxa = [tree2.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    # Get intersection - this is what lattice construction should use
    intersection = splits1.intersection(splits2)
    print(f"\nIntersection (common splits): {len(intersection)}")
    for split in sorted(intersection, key=lambda x: x.indices):
        taxa = [tree1.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    # Now check what lattice construction actually finds
    print(f"\n" + "="*60)
    print("LATTICE CONSTRUCTION RESULTS")
    print("="*60)
    
    # Call lattice construction directly
    lattice_edges = construct_sub_lattices(tree1, tree2)
    
    print(f"Lattice construction found {len(lattice_edges)} edges:")
    for edge in lattice_edges:
        taxa = [tree1.get_leaves()[i].name for i in edge.split.indices]
        print(f"  {edge.split.indices} -> {taxa}")
        
        # Check if this edge is actually in the intersection
        if edge.split in intersection:
            print(f"    ✓ This edge IS in the intersection (common splits)")
        else:
            print(f"    ✗ This edge is NOT in the intersection!")
            print(f"      In Tree 1: {edge.split in splits1}")
            print(f"      In Tree 2: {edge.split in splits2}")
    
    # Specifically check our problematic split
    problematic = Partition((2, 3, 4, 5, 6, 7, 8, 9))
    print(f"\n" + "="*60)
    print(f"PROBLEMATIC SPLIT ANALYSIS: {problematic.indices}")
    print("="*60)
    
    print(f"Split {problematic.indices}:")
    print(f"  In Tree 1 splits: {problematic in splits1}")
    print(f"  In Tree 2 splits: {problematic in splits2}")
    print(f"  In intersection: {problematic in intersection}")
    
    # Check if lattice construction found it
    found_in_lattice = any(edge.split == problematic for edge in lattice_edges)
    print(f"  Found by lattice construction: {found_in_lattice}")
    
    if found_in_lattice:
        print(f"  🚨 BUG: Lattice construction found a split that's not in both trees!")
    else:
        print(f"  ✓ Lattice construction correctly ignored this split")
    
    # The question is: where is this s-edge coming from then?
    print(f"\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    print("If lattice construction is working correctly (only finding common splits),")
    print("then the problematic s-edge must be coming from somewhere else.")
    print("Possible sources:")
    print("1. The iterate_lattice_algorithm function does additional processing")
    print("2. There's a bug in split comparison/hashing")
    print("3. The s-edge is generated during the iterative algorithm process")
    print("4. There's caching causing stale data")


if __name__ == "__main__":
    test_lattice_debug_detailed()