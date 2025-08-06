#!/usr/bin/env python3
"""Test to capture details about the failed s-edge."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.interpolation import build_lattice_interpolation_sequence
from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm


def test_failed_s_edge():
    """Identify the failed s-edge and understand why it failed."""
    # The exact trees from the original request
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    print("ANALYZING FAILED S-EDGE")
    print("="*50)
    
    # Parse with correct encoding
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick, order=tree1._order)
    
    # Get the lattice edge solutions directly
    lattice_edge_solutions = iterate_lattice_algorithm(tree1, tree2)
    
    print(f"\nFound {len(lattice_edge_solutions)} s-edges from lattice algorithm:")
    
    for i, (s_edge, solutions) in enumerate(lattice_edge_solutions.items()):
        taxa_names = [tree1.get_leaves()[j].name for j in s_edge.indices]
        print(f"\n{i+1}. S-edge: {s_edge.indices}")
        print(f"   Taxa: {taxa_names}")
        print(f"   Solutions: {len(solutions)} solution sets")
        
        for j, solution_set in enumerate(solutions):
            print(f"   Solution set {j+1}:")
            for partition in solution_set:
                part_taxa = [tree1.get_leaves()[k].name for k in partition.indices]
                print(f"     - {partition.indices} -> {part_taxa}")
    
    # Now run the interpolation to see which one fails
    print("\n" + "="*50)
    print("RUNNING INTERPOLATION TO IDENTIFY FAILED S-EDGE")
    print("="*50)
    
    try:
        # This will show which s-edge fails during processing
        result = build_lattice_interpolation_sequence(tree1, tree2, 0)
        
        print(f"\nInterpolation completed successfully!")
        print(f"Generated {len(result.trees)} trees")
        
        # Compare lattice solutions with what was actually processed
        print(f"\nOriginal s-edges found: {len(lattice_edge_solutions)}")
        print(f"S-edges in result: {len(result.lattice_edge_solutions)}")
        
        # Show which s-edges were successfully processed
        successful_s_edges = set(result.lattice_edge_solutions.keys())
        all_s_edges = set(lattice_edge_solutions.keys())
        
        print(f"\nSuccessfully processed s-edges:")
        for s_edge in successful_s_edges:
            taxa = [tree1.get_leaves()[j].name for j in s_edge.indices]
            print(f"  ✓ {s_edge.indices} -> {taxa}")
        
        failed_s_edges = all_s_edges - successful_s_edges
        print(f"\nFailed s-edges:")
        for s_edge in failed_s_edges:
            taxa = [tree1.get_leaves()[j].name for j in s_edge.indices]
            print(f"  ✗ {s_edge.indices} -> {taxa}")
            
            # Check why this s-edge failed
            print(f"    Checking if s-edge exists in both trees:")
            
            node1 = tree1.find_node_by_split(s_edge)
            node2 = tree2.find_node_by_split(s_edge)
            
            if node1:
                taxa1 = sorted([leaf.name for leaf in node1.get_leaves()])
                print(f"      Tree 1: ✓ Found -> {taxa1}")
            else:
                print(f"      Tree 1: ✗ Not found")
            
            if node2:
                taxa2 = sorted([leaf.name for leaf in node2.get_leaves()])
                print(f"      Tree 2: ✓ Found -> {taxa2}")
            else:
                print(f"      Tree 2: ✗ Not found")
                
            # This is likely why it failed - the s-edge doesn't exist in both trees
            if not (node1 and node2):
                print(f"    → REASON FOR FAILURE: S-edge not found in both trees")
    
    except Exception as e:
        print(f"Error during interpolation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_failed_s_edge()