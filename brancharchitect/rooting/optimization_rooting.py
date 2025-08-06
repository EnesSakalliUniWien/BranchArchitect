"""
Advanced optimization and matching algorithms for phylogenetic tree rerooting.

This module provides sophisticated rerooting strategies including:
- Jaccard similarity-based matching
- Global correspondence mapping
- Enhanced optimization algorithms (phylo-io inspired)
- Complex node matching and correspondence

Author: BranchArchitect Team
"""

from typing import Optional, Dict, Tuple
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .core_rooting import reroot_at_node
from .root_selection import find_optimal_root_candidates
from .optimization_utilities import (
    fallback_to_simple_rerooting,
    validate_and_rebuild_tree_structure,
    select_best_non_leaf_candidate,
)


def reroot_to_compared_tree(
    tree1: Node,
    tree2: Node,
    splits1: Optional[PartitionSet[Partition]] = None,
    splits2: Optional[PartitionSet[Partition]] = None,
    similarity_matrix: Optional[Dict[Tuple[Partition, Partition], float]] = None,
) -> Node:
    """
    Reroot tree1 to maximize similarity with tree2 using advanced optimization.

    Args:
        tree1: Tree to reroot
        tree2: Reference tree
        splits1: Precomputed splits for tree1 (optional)
        splits2: Precomputed splits for tree2 (optional)
        similarity_matrix: Precomputed similarity matrix (optional)

    Returns:
        Optimally rerooted tree1
    """
    # Fallback for missing data
    if not splits2 or not similarity_matrix:
        return fallback_to_simple_rerooting(tree1, tree2)

    try:
        # Find optimal root candidates
        candidates = find_optimal_root_candidates(tree1, splits2, similarity_matrix)

        if not candidates:
            return fallback_to_simple_rerooting(tree1, tree2)

        # Select best non-leaf candidate
        best_candidate: Node | None = select_best_non_leaf_candidate(candidates)

        if best_candidate is None:
            # Use first candidate if no non-leaf candidates
            best_candidate = candidates[0][0]

        # Reroot at the selected candidate
        rerooted_tree: Node = reroot_at_node(best_candidate)

        # Validate and return
        return validate_and_rebuild_tree_structure(rerooted_tree)

    except Exception:
        # Fallback on any error
        return fallback_to_simple_rerooting(tree1, tree2)
