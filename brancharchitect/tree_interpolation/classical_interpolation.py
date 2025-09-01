"""
Core tree calculation functions for interpolation.

This module contains the fundamental algorithms for calculating
intermediate and consensus trees during the interpolation process.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.consensus_tree.consensus_tree import (
    calculate_consensus_tree,
)
from brancharchitect.tree_interpolation.consensus_tree.intermediate_tree import (
    calculate_intermediate_tree,
)


def classical_interpolation(
    target: Node,
    reference: Node,
    split_data: Tuple[Dict[Partition, float], Dict[Partition, float]],
) -> List[Node]:
    """Create consensus tree sequence and mappings."""
    split_dict1, split_dict2 = split_data

    # Create intermediate and consensus trees
    it1: Node = calculate_intermediate_tree(target, split_dict2)
    it2: Node = calculate_intermediate_tree(reference, split_dict1)
    c1: Node = calculate_consensus_tree(it1, split_dict2)
    c2: Node = calculate_consensus_tree(it2, split_dict1)

    return [it1, c1, c2, it2]
