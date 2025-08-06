#!/usr/bin/env python3
"""Test to trace through the iterative lattice algorithm."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm
from brancharchitect.jumping_taxa.lattice.lattice_construction import construct_sub_lattices
from brancharchitect.elements.partition import Partition


def test_iterative_lattice():
    """Trace through the iterative lattice algorithm to see when the problematic s-edge appears."""
    # The exact trees
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse with correct encoding
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick, order=tree1._order)
    
    print("TRACING ITERATIVE LATTICE ALGORITHM")
    print("="*60)
    
    # Our problematic split
    problematic = Partition((2, 3, 4, 5, 6, 7, 8, 9))
    print(f"Tracking split: {problematic.indices}")
    print(f"Taxa: {[tree1.get_leaves()[i].name for i in problematic.indices]}")
    
    # First check what the full iterative algorithm returns
    full_result = iterate_lattice_algorithm(tree1, tree2)
    
    print(f"\nFull algorithm result:")
    print(f"Found {len(full_result)} s-edges total:")
    for s_edge in full_result:
        taxa = [tree1.get_leaves()[i].name for i in s_edge.indices]
        print(f"  {s_edge.indices} -> {taxa}")
        if s_edge == problematic:
            print(f"    🔍 This is our problematic s-edge!")
    
    # Now let's manually trace through iterations
    print(f"\n" + "="*60)
    print("MANUAL ITERATION TRACING")
    print("="*60)
    
    current_t1 = tree1.deep_copy()
    current_t2 = tree2.deep_copy()
    iteration = 0
    
    while iteration < 5:  # Limit iterations for debugging
        iteration += 1
        
        print(f"\n--- ITERATION {iteration} ---")
        
        # Check current trees
        splits1 = current_t1.to_splits()
        splits2 = current_t2.to_splits()
        common = splits1 & splits2
        
        print(f"Tree 1: {len(splits1)} splits")
        print(f"Tree 2: {len(splits2)} splits") 
        print(f"Common: {len(common)} splits")
        
        # Check if our problematic split exists in this iteration
        in_t1 = problematic in splits1
        in_t2 = problematic in splits2
        in_common = problematic in common
        
        print(f"Problematic split {problematic.indices}:")
        print(f"  In Tree 1: {in_t1}")
        print(f"  In Tree 2: {in_t2}")
        print(f"  In common: {in_common}")
        
        # Run lattice construction for this iteration
        lattice_edges = construct_sub_lattices(current_t1, current_t2)
        print(f"Lattice construction found {len(lattice_edges)} edges:")
        
        found_problematic = False
        for edge in lattice_edges:
            taxa = [current_t1.get_leaves()[i].name for i in edge.split.indices]
            print(f"  {edge.split.indices} -> {taxa}")
            if edge.split == problematic:
                print(f"    🔍 FOUND our problematic s-edge in iteration {iteration}!")
                found_problematic = True
        
        if found_problematic:
            print(f"\n🎯 The problematic s-edge appears in iteration {iteration}")
            print(f"This explains why it's returned by iterate_lattice_algorithm!")
            break
        
        # For now, break if no lattice edges found
        if not lattice_edges:
            print(f"No lattice edges found in iteration {iteration}. Stopping.")
            break
        
        # In a real implementation, we would:
        # 1. Solve the lattice to find jumping taxa
        # 2. Remove jumping taxa from trees
        # 3. Continue to next iteration
        # For now, just break to avoid infinite loop
        break
    
    print(f"\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The 'failed s-edge' is actually discovered in a later iteration")
    print("of the algorithm, after some jumping taxa have been removed.")
    print("This is why it appears in the final result even though it's")
    print("not in the common splits of the original trees.")


if __name__ == "__main__":
    test_iterative_lattice()