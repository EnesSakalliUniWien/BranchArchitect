#!/usr/bin/env python3
"""Analyze why (2, 3, 4, 5, 6, 7, 8, 9) is an s-edge when it doesn't exist in both trees."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm
from brancharchitect.elements.partition import Partition


def test_s_edge_analysis():
    """Analyze the s-edge logic."""
    # The exact trees
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse with correct encoding
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick, order=tree1._order)
    
    print("COMPARING SPLITS BETWEEN TREES")
    print("="*60)
    
    # Get all splits
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()
    
    print(f"Tree 1 has {len(splits1)} splits")
    print(f"Tree 2 has {len(splits2)} splits")
    
    # Find common and unique splits
    common_splits = splits1 & splits2
    unique_to_tree1 = splits1 - splits2
    unique_to_tree2 = splits2 - splits1
    
    print(f"\nCommon splits: {len(common_splits)}")
    for split in sorted(common_splits, key=lambda x: x.indices):
        taxa = [tree1.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    print(f"\nUnique to Tree 1: {len(unique_to_tree1)}")
    for split in sorted(unique_to_tree1, key=lambda x: x.indices):
        taxa = [tree1.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    print(f"\nUnique to Tree 2: {len(unique_to_tree2)}")
    for split in sorted(unique_to_tree2, key=lambda x: x.indices):
        taxa = [tree2.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    # Now check what the lattice algorithm finds
    print("\n" + "="*60)
    print("LATTICE ALGORITHM RESULTS")
    print("="*60)
    
    lattice_edge_solutions = iterate_lattice_algorithm(tree1, tree2)
    
    print(f"Lattice algorithm found {len(lattice_edge_solutions)} s-edges:")
    for s_edge, solutions in lattice_edge_solutions.items():
        taxa = [tree1.get_leaves()[i].name for i in s_edge.indices]
        print(f"\nS-edge: {s_edge.indices} -> {taxa}")
        
        # Check if this s-edge exists in both trees
        in_tree1 = s_edge in splits1
        in_tree2 = s_edge in splits2
        print(f"  In Tree 1: {in_tree1}")
        print(f"  In Tree 2: {in_tree2}")
        
        if not (in_tree1 and in_tree2):
            print(f"  ⚠️  WARNING: This s-edge is not in both trees!")
            
            if in_tree1 and not in_tree2:
                print(f"     This split exists in Tree 1 but not Tree 2")
                print(f"     Tree 2 closest matches:")
                # Find Tree 2 splits that contain some of these taxa
                s_edge_taxa = set([tree1.get_leaves()[i].name for i in s_edge.indices])
                for t2_split in splits2:
                    t2_taxa = set([tree2.get_leaves()[i].name for i in t2_split.indices])
                    overlap = s_edge_taxa.intersection(t2_taxa)
                    if len(overlap) >= len(s_edge_taxa) - 2:  # Close matches
                        print(f"       {t2_split.indices} -> {sorted(t2_taxa)} (missing: {s_edge_taxa - t2_taxa}, extra: {t2_taxa - s_edge_taxa})")
        
        print(f"  Solutions: {len(solutions)} sets")
        for i, solution_set in enumerate(solutions):
            print(f"    Set {i+1}: {[sol.indices for sol in solution_set]} -> {[[tree1.get_leaves()[j].name for j in sol.indices] for sol in solution_set]}")
    
    # The key question: How can something be an s-edge if it doesn't exist in both trees?
    print("\n" + "="*60)
    print("ANALYSIS: WHY IS THIS AN S-EDGE?")
    print("="*60)
    
    print("The lattice algorithm identifies s-edges based on the DIFFERENCES between trees.")
    print("An s-edge represents a structural difference that needs to be resolved during interpolation.")
    print("")
    print("The failed s-edge (2, 3, 4, 5, 6, 7, 8, 9) represents:")
    print("• A grouping that EXISTS in Tree 1")
    print("• But does NOT exist in Tree 2") 
    print("• Tree 2 has a similar but LARGER grouping that includes B2")
    print("")
    print("This IS a legitimate structural difference that the algorithm detected!")
    print("The algorithm tries to interpolate this difference but fails because:")
    print("• It cannot find the target split in Tree 2 to determine the ordering")
    print("• So it falls back to classical interpolation")


if __name__ == "__main__":
    test_s_edge_analysis()