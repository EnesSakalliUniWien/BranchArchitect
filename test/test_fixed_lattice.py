#!/usr/bin/env python3
"""Test the fixed lattice algorithm that filters s-edges."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm
from brancharchitect.tree_interpolation.interpolation import build_sequential_lattice_interpolations


def test_fixed_lattice():
    """Test the fixed lattice algorithm."""
    # The exact trees
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse with correct encoding
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick, order=tree1._order)
    
    print("TESTING FIXED LATTICE ALGORITHM")
    print("="*60)
    
    # Check original common splits
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()
    original_common = splits1.intersection(splits2)
    
    print(f"Original common splits: {len(original_common)}")
    for split in sorted(original_common, key=lambda x: x.indices):
        taxa = [tree1.get_leaves()[i].name for i in split.indices]
        print(f"  {split.indices} -> {taxa}")
    
    # Test the fixed lattice algorithm
    print(f"\n" + "="*60)
    print("FIXED LATTICE ALGORITHM RESULTS")
    print("="*60)
    
    result = iterate_lattice_algorithm(tree1, tree2)
    
    print(f"Fixed algorithm returned {len(result)} s-edges:")
    for s_edge, solutions in result.items():
        taxa = [tree1.get_leaves()[i].name for i in s_edge.indices]
        print(f"  {s_edge.indices} -> {taxa}")
        print(f"    Solutions: {len(solutions)} sets")
    
    # Check if problematic s-edge is filtered out
    from brancharchitect.elements.partition import Partition
    problematic = Partition((2, 3, 4, 5, 6, 7, 8, 9))
    
    if problematic in result:
        print(f"\n❌ Problematic s-edge {problematic.indices} still present!")
    else:
        print(f"\n✅ Problematic s-edge {problematic.indices} successfully filtered out!")
    
    # Test interpolation with the fixed algorithm
    print(f"\n" + "="*60)
    print("TESTING INTERPOLATION WITH FIXED ALGORITHM")
    print("="*60)
    
    tree_list = [tree1, tree2]
    
    try:
        interpolation_result = build_sequential_lattice_interpolations(tree_list)
        print("✅ Interpolation succeeded!")
        print(f"Generated {len(interpolation_result.interpolated_trees)} total trees")
        
        # Check if classical interpolation fallback was used
        classical_count = sum(1 for name in interpolation_result.interpolation_sequence_labels 
                            if 'classical' in name)
        lattice_count = len(interpolation_result.interpolated_trees) - 2 - classical_count  # Subtract original trees
        
        print(f"Classical interpolation steps: {classical_count}")
        print(f"Lattice-based interpolation steps: {lattice_count}")
        
        if classical_count == 0:
            print("🎉 No classical fallback needed - all s-edges can be interpolated!")
        else:
            print(f"⚠️  {classical_count} classical fallback steps still needed")
            
    except Exception as e:
        print(f"❌ Interpolation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_lattice()