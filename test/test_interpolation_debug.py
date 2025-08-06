#!/usr/bin/env python3
"""Debug test to trace the interpolation error."""

import pytest
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.interpolation import build_sequential_lattice_interpolations
from brancharchitect.tree_interpolation.helpers import generate_s_edge_interpolation_sequence, _reorder_consensus_tree_by_edge
import brancharchitect.tree_interpolation.helpers as helpers


# Monkey patch the reorder function to add debugging
original_reorder = _reorder_consensus_tree_by_edge

def debug_reorder_consensus_tree_by_edge(consensus_tree, target_tree, edge):
    """Debug wrapper for _reorder_consensus_tree_by_edge."""
    print(f"\n=== DEBUG: Entering _reorder_consensus_tree_by_edge ===")
    print(f"Edge: {edge.indices}")
    print(f"\nConsensus tree structure:")
    print(consensus_tree.to_newick())
    consensus_taxa = sorted([leaf.name for leaf in consensus_tree.get_leaves()])
    print(f"Consensus tree taxa: {consensus_taxa}")
    
    print(f"\nTarget tree structure:")
    print(target_tree.to_newick())
    target_taxa = sorted([leaf.name for leaf in target_tree.get_leaves()])
    print(f"Target tree taxa: {target_taxa}")
    
    # Try to find the edge node in each tree
    print(f"\nSearching for edge {edge.indices} in trees...")
    
    consensus_node = consensus_tree.find_node_by_split(edge)
    if consensus_node:
        consensus_node_taxa = sorted([leaf.name for leaf in consensus_node.get_leaves()])
        print(f"Found in consensus tree: Node with taxa {consensus_node_taxa}")
    else:
        print("NOT found in consensus tree")
    
    target_node = target_tree.find_node_by_split(edge)
    if target_node:
        target_node_taxa = sorted([leaf.name for leaf in target_node.get_leaves()])
        print(f"Found in target tree: Node with taxa {target_node_taxa}")
        print(f"Target node current order: {list(target_node.get_current_order())}")
    else:
        print("NOT found in target tree")
    
    print("\n=== Calling original reorder function ===")
    
    try:
        result = original_reorder(consensus_tree, target_tree, edge)
        print("Reorder successful!")
        return result
    except Exception as e:
        print(f"Reorder failed with error: {e}")
        raise

# Apply the monkey patch
helpers._reorder_consensus_tree_by_edge = debug_reorder_consensus_tree_by_edge


def test_debug_interpolation():
    """Test with debugging to trace the error."""
    # Define the two trees in Newick format
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse the trees
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick)
    
    print("Tree 1 (Target):")
    print(tree1.to_newick())
    print("\nTree 2 (Reference):")
    print(tree2.to_newick())
    
    # Get taxa from both trees
    taxa1 = sorted([leaf.name for leaf in tree1.get_leaves()])
    taxa2 = sorted([leaf.name for leaf in tree2.get_leaves()])
    
    print(f"\nTree 1 taxa: {taxa1}")
    print(f"Tree 2 taxa: {taxa2}")
    print(f"Taxa match: {taxa1 == taxa2}")
    
    # Create a list of trees for interpolation
    tree_list = [tree1, tree2]
    
    try:
        # Run the interpolation
        print("\n" + "="*60)
        print("Starting interpolation...")
        print("="*60)
        
        result = build_sequential_lattice_interpolations(tree_list)
        print(f"\nInterpolation successful!")
        print(f"Generated {len(result.interpolated_trees)} trees")
        
    except ValueError as e:
        print(f"\n{'='*60}")
        print(f"Interpolation failed with ValueError:")
        print(f"{'='*60}")
        print(f"Error: {e}")
        
        if len(e.args) == 3 and "Permutation must include all taxa" in str(e.args[0]):
            print(f"\nExpected taxa (permutation): {e.args[1]}")
            print(f"Actual taxa in tree: {e.args[2]}")
            print(f"Missing from tree: {set(e.args[1]) - e.args[2]}")
            print(f"Extra in tree: {e.args[2] - set(e.args[1])}")


if __name__ == "__main__":
    test_debug_interpolation()