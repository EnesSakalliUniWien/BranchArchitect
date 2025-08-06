from typing import List
from brancharchitect.io import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import (
    iterate_lattice_algorithm,
)
from brancharchitect.tree import Node
from brancharchitect.rooting.rooting import simple_reroot
from brancharchitect.plot.tree_printer import print_tree_comparison

t1: Node = Node()
t2: Node = Node()

trees: Node | List[Node] = parse_newick(
    ("(O,(((C1,Y),((B2,C2),B1)),X));" + "(O,(((B2,Y),((B1,X),C2)),C1));")
)
t1, t2 = trees[0], trees[1]


print_tree_comparison(t1, t2)
t2: Node = simple_reroot(t2, t1)

print_tree_comparison(t1, t2)


s_edge_solutions = iterate_lattice_algorithm(t1, t2, t1._order)

print(s_edge_solutions)
print("Lattice algorithm completed.")
print("Lattice algorithm completed.")
"""
from typing import Dict, List, Set
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node


def interpolate_tree_with_s_edge_depth(tree_one: Node, tree_two: Node) -> List[Node]:
    Create fine-grained interpolation using s_edge_depth for progressive subtree ordering.

    Each step cumulatively adds ordering - preserving previous orderings and adding one more depth level.

    Args:
        tree_one: First input tree
        tree_two: Second input tree

    Returns:
        List of trees showing smooth progression with cumulative ordering
    # Get the standard interpolation components
    split_dict1: Dict[Partition, float] = tree_one.to_weighted_splits()
    split_dict2: Dict[Partition, float] = tree_two.to_weighted_splits()

    it1: Node = calculate_intermediate_tree(tree_one, split_dict2)
    it2: Node = calculate_intermediate_tree(tree_two, split_dict1)

    # Collect all unique s_edge_depth values from both trees
    all_depths: Set[int] = set()
    _collect_s_edge_depths(tree_one, all_depths)
    _collect_s_edge_depths(tree_two, all_depths)

    # Sort depths from highest to lowest for progressive ordering
    sorted_depths = sorted(all_depths, reverse=True)

    # Create incremental consensus trees with CUMULATIVE ordering
    incremental_trees: List[Node] = []

    # Generate consensus trees with progressively more depth levels ordered
    for i in range(len(sorted_depths) + 1):
        # CUMULATIVE: Include all depths from 0 to i (not just depth i)
        depths_to_order = set(sorted_depths[:i]) if i > 0 else set()

        # Create consensus tree with cumulative ordering
        c1_incremental = _calculate_cumulative_consensus_tree(
            it1.deep_copy(), split_dict2, depths_to_order
        )
        c2_incremental = _calculate_cumulative_consensus_tree(
            it2.deep_copy(), split_dict1, depths_to_order
        )

        incremental_trees.extend([c1_incremental, c2_incremental])

    # Return the complete sequence
    result = [tree_one, it1]
    result.extend(incremental_trees)
    result.extend([it2, tree_two])

    return result


def _calculate_cumulative_consensus_tree(
    node: Node, split_dict: Dict[Partition, float], depths_to_order: Set[int]
) -> Node:
    Calculate consensus tree with cumulative ordering based on s_edge_depth.

    All nodes with s_edge_depth in depths_to_order will have their children sorted.
    This preserves ordering from previous steps while adding new ordering levels.
    # If the node is a leaf, return it unchanged.
    if not node.children:
        return node

    new_children: list[Node] = []
    for child in node.children:
        # Recursively process the child with cumulative ordering
        processed_child: Node = _calculate_cumulative_consensus_tree(
            child, split_dict, depths_to_order
        )
        # If the processed child is internal (has children)
        if processed_child.children:
            # If its split is in the consensus splits, keep the whole node.
            if processed_child.split_indices in split_dict:
                new_children.append(processed_child)
            else:
                # Otherwise, collapse it by promoting its children.
                for grandchild in processed_child.children:
                    new_children.append(grandchild)
        else:
            # Leaf nodes are always kept.
            new_children.append(processed_child)

    # Apply CUMULATIVE ordering: sort if this node's depth is in the cumulative set
    if (
        hasattr(node, "s_edge_depth")
        and node.s_edge_depth in depths_to_order
        and new_children
    ):
        # Sort children by s_edge_depth (highest first)
        new_children.sort(key=lambda n: getattr(n, "s_edge_depth", 0), reverse=True)

    node.children = new_children
    return node
"""
