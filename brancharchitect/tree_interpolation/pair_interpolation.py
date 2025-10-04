"""
Active-changing split interpolation pipeline.

Builds the interpolation sequence between two trees by identifying active-changing
splits (formerly called s-edges or lattice edges) via the lattice algorithm and
generating a five-step morphing sequence per active-changing split.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths import (
    create_interpolation_for_active_split_sequence,
)

# Import here to avoid circular import
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.tree_interpolation.types import (
    TreePairInterpolation,
    LatticeEdgeData,
)
# distance_metrics import removed (metrics no longer used here)

logger: logging.Logger = logging.getLogger(__name__)

__all__: List[str] = [
    "process_tree_pair_interpolation",
]


def discover_active_changing_splits(
    source_tree: Node,
    destination_tree: Node,
    precomputed_solutions: Optional[Dict[Partition, List[List[Partition]]]] = None,
) -> LatticeEdgeData:
    """
    Discover active-changing splits between two trees using the lattice algorithm.

    Args:
        source_tree: Source tree for interpolation
        destination_tree: Destination tree for interpolation

    Returns:
        LatticeEdgeData containing discovered edges and their jumping subtree solutions
    """
    # The caller (SequentialInterpolationBuilder) already passes deep-copied trees
    # for each pair, so we can call the lattice algorithm directly without
    # performing additional copies here.
    # Use precomputed lattice solutions when provided; otherwise compute fresh
    jumping_subtree_solutions: Dict[Partition, List[List[Partition]]] = (
        precomputed_solutions
        if precomputed_solutions is not None
        else iterate_lattice_algorithm(source_tree, destination_tree)
    )

    lattice_edges: List[Partition] = list(jumping_subtree_solutions.keys())
    active_split_data = LatticeEdgeData(lattice_edges, jumping_subtree_solutions)
    active_split_data.compute_depths(source_tree, destination_tree)

    return active_split_data


def process_tree_pair_interpolation(
    source_tree: Node,
    destination_tree: Node,
    precomputed_solutions: Optional[Dict[Partition, List[List[Partition]]]] = None,
) -> TreePairInterpolation:
    """
    Build interpolation sequence using active-changing splits.

    Pipeline:
    1. Discover active-changing splits via lattice algorithm
    2. Order splits by depth (leaves to root)
    3. Generate solution-to-atom mappings
    4. Create 5-step morphing sequence per active-changing split
    5. Calculate quality metrics
    6. Assemble final result

    The 5-step sequence for each active-changing split:
    - IT{i}_down_{j}: Apply reference weights to target topology
    - C{i}_{j}: Collapse zero-length branches
    - C{i}_{j}_reorder: Match reference node ordering
    - IT{i}_up_{j}: Reference topology with target weights
    - IT{i}_ref_{j}: Complete transformation with reference weights

    Args:
        source_tree: Starting tree for interpolation
        destination_tree: Destination tree for interpolation
        precomputed_solutions: Pre-computed lattice solutions (optional)

    Returns:
        TreePairInterpolation aggregating trees, mappings, tracking, and metrics
    """
    # Step 1: Discover active-changing splits between trees
    active_split_data: LatticeEdgeData = discover_active_changing_splits(
        source_tree, destination_tree, precomputed_solutions
    )

    # Step 2: Order splits by depth for optimal interpolation progression
    ordered_edges = active_split_data.get_sorted_edges(
        use_reference=False, ascending=True
    )

    sequence_trees, failed_active_split, active_split_changing_tracking = (
        create_interpolation_for_active_split_sequence(
            source_tree=source_tree,
            destination_tree=destination_tree,
            target_active_changing_edges=ordered_edges,
            jumping_subtree_solutions=active_split_data.jumping_subtree_solutions,
        )
    )

    # Log any failed edges for debugging
    if failed_active_split:
        logger.warning(
            f"Classical interpolation used for {len(failed_active_split)} failed active-changing splits"
        )

    return TreePairInterpolation(
        trees=sequence_trees,
        active_changing_split_tracking=active_split_changing_tracking,
        jumping_subtree_solutions=active_split_data.jumping_subtree_solutions,
    )
