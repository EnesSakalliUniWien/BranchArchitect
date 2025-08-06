#!/usr/bin/env python3
"""Detailed trace of the reordering error."""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.interpolation import build_sequential_lattice_interpolations
from brancharchitect.tree_interpolation.helpers import _reorder_consensus_tree_by_edge
from brancharchitect.elements.partition import Partition
import brancharchitect.tree_interpolation.helpers as helpers


# Store intermediate trees for analysis
captured_trees = {}

# Monkey patch to capture the consensus tree before reordering
original_reorder = _reorder_consensus_tree_by_edge

def capture_reorder(consensus_tree, target_tree, edge):
    """Capture trees before the error."""
    global captured_trees
    captured_trees['consensus'] = consensus_tree
    captured_trees['target'] = target_tree
    captured_trees['edge'] = edge
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("REORDER FUNCTION CALLED")
    print("="*60)
    
    print(f"\nEdge being processed: {edge.indices}")
    
    print("\n--- CONSENSUS TREE ---")
    print(f"Full tree: {consensus_tree.to_newick()}")
    consensus_leaves = sorted([leaf.name for leaf in consensus_tree.get_leaves()])
    print(f"All leaves: {consensus_leaves}")
    
    # Find the node with this edge
    consensus_node = consensus_tree.find_node_by_split(edge)
    if consensus_node:
        node_leaves = sorted([leaf.name for leaf in consensus_node.get_leaves()])
        print(f"Node with edge {edge.indices} contains: {node_leaves}")
    else:
        print(f"Node with edge {edge.indices}: NOT FOUND")
    
    print("\n--- TARGET TREE ---")
    print(f"Full tree: {target_tree.to_newick()}")
    target_leaves = sorted([leaf.name for leaf in target_tree.get_leaves()])
    print(f"All leaves: {target_leaves}")
    
    # Find the node with this edge
    target_node = target_tree.find_node_by_split(edge)
    if target_node:
        node_leaves = sorted([leaf.name for leaf in target_node.get_leaves()])
        print(f"Node with edge {edge.indices} contains: {node_leaves}")
        print(f"Node's current order: {list(target_node.get_current_order())}")
    else:
        print(f"Node with edge {edge.indices}: NOT FOUND")
    
    print("\n--- LEAF COMPARISON ---")
    print(f"Do both trees have same leaves? {consensus_leaves == target_leaves}")
    
    if consensus_leaves != target_leaves:
        print(f"Leaves only in consensus: {set(consensus_leaves) - set(target_leaves)}")
        print(f"Leaves only in target: {set(target_leaves) - set(consensus_leaves)}")
    
    # Now call the original function to see the error
    return original_reorder(consensus_tree, target_tree, edge)

helpers._reorder_consensus_tree_by_edge = capture_reorder


def test_error_trace():
    """Trace the exact error."""
    # Define the two trees
    tree1_newick = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_newick = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    
    # Parse the trees
    tree1 = parse_newick(tree1_newick)
    tree2 = parse_newick(tree2_newick)
    
    print("INITIAL TREES")
    print("="*60)
    
    leaves1 = sorted([leaf.name for leaf in tree1.get_leaves()])
    leaves2 = sorted([leaf.name for leaf in tree2.get_leaves()])
    
    print(f"\nTree 1 leaves: {leaves1}")
    print(f"Tree 2 leaves: {leaves2}")
    print(f"Same leaves? {leaves1 == leaves2}")
    
    # Check split indices initialization
    print("\n" + "="*60)
    print("CHECKING SPLIT INDEX INITIALIZATION")
    print("="*60)
    
    # Check if splits are initialized
    print(f"\nTree 1 has split indices? {hasattr(tree1, '_split_index')}")
    print(f"Tree 2 has split indices? {hasattr(tree2, '_split_index')}")
    
    # Initialize split indices
    print("\nInitializing split indices...")
    # Create encoding from tree1's taxa
    taxa_names = [leaf.name for leaf in tree1.get_leaves()]
    encoding = {name: i for i, name in enumerate(taxa_names)}
    
    tree1.initialize_split_indices(encoding)
    tree2.initialize_split_indices(encoding)
    
    # Check leaf ordering after initialization
    print("\nLeaf ordering after split index initialization:")
    leaf_names1 = [leaf.name for leaf in tree1.get_leaves()]
    leaf_names2 = [leaf.name for leaf in tree2.get_leaves()]
    
    print(f"Tree 1 leaf order: {[f'{i}:{name}' for i, name in enumerate(leaf_names1)]}")
    print(f"Tree 2 leaf order: {[f'{i}:{name}' for i, name in enumerate(leaf_names2)]}")
    
    # Check if the orders are the same
    print(f"\nSame leaf ordering? {leaf_names1 == leaf_names2}")
    if leaf_names1 != leaf_names2:
        print("WARNING: Trees have different leaf orderings!")
        for i, (n1, n2) in enumerate(zip(leaf_names1, leaf_names2)):
            if n1 != n2:
                print(f"  Position {i}: Tree1={n1}, Tree2={n2}")
    
    # Check specific partitions
    print("\n" + "="*60)
    print("CHECKING SPECIFIC PARTITIONS")
    print("="*60)
    
    # Get all splits from both trees
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()
    
    # Check partition (2,3,4,5)
    test_partition = Partition((2, 3, 4, 5))
    print(f"\nChecking partition {test_partition.indices}:")
    
    if test_partition in splits1:
        print(f"  In Tree 1: YES -> {[leaf_names1[i] for i in test_partition.indices]}")
    else:
        print(f"  In Tree 1: NO")
    
    if test_partition in splits2:
        print(f"  In Tree 2: YES -> {[leaf_names2[i] for i in test_partition.indices]}")
    else:
        print(f"  In Tree 2: NO")
    
    # Create tree list
    tree_list = [tree1, tree2]
    
    try:
        # Run the interpolation to trigger the error
        print("\n" + "="*60)
        print("STARTING INTERPOLATION")
        print("="*60)
        
        result = build_sequential_lattice_interpolations(tree_list)
        print("Interpolation succeeded unexpectedly!")
        
    except ValueError as e:
        print("\n" + "="*60)
        print("ERROR CAUGHT")
        print("="*60)
        print(f"\nError: {e}")
        
        if len(e.args) == 3:
            print(f"\nPermutation requested: {e.args[1]}")
            print(f"Taxa in node: {e.args[2]}")
            print(f"Missing taxa: {set(e.args[1]) - e.args[2]}")
            print(f"Extra taxa: {e.args[2] - set(e.args[1])}")
        
        # Analyze captured trees
        if captured_trees:
            print("\n" + "="*60)
            print("ANALYSIS OF CAPTURED STATE")
            print("="*60)
            
            consensus = captured_trees.get('consensus')
            target = captured_trees.get('target')
            edge = captured_trees.get('edge')
            
            if consensus and target and edge:
                # Check if the trees are what we expect
                print(f"\nConsensus tree is based on Tree 1? (checking structure)")
                print(f"Target tree is Tree 2? (checking structure)")
                
                # The key question: why does the target tree have a node with B2 
                # when the consensus tree has C2?
                print("\n--- THE KEY QUESTION ---")
                print("Why is the algorithm trying to use an order containing B2")
                print("when the consensus tree node contains C2?")
                
                # Check the reference tree used for interpolation
                print("\n--- CHECKING REFERENCE WEIGHTS ---")
                print("The consensus tree was created by collapsing certain branches.")
                print("During the reorder step, it's using the REFERENCE tree (tree2)")
                print("to determine the order, not the consensus tree!")


if __name__ == "__main__":
    test_error_trace()