"""
Active-changing split interpolation pipeline.

Builds the interpolation sequence between two trees by identifying active-changing
splits (formerly called s-edges or lattice edges) via the lattice algorithm and
generating a five-step morphing sequence per active-changing split.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths import (
    orchestrate_active_split_sequence,
)

# Import here to avoid circular import
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.jumping_taxa.lattice.mapping import map_solutions_to_atoms
from brancharchitect.tree_interpolation.types import (
    TreePairInterpolation,
    LatticeEdgeData,
)
# distance_metrics import removed (metrics no longer used here)

logger: logging.Logger = logging.getLogger(__name__)

__all__: List[str] = [
    "build_active_changing_split_interpolation_sequence",
]


def discover_active_changing_splits(target: Node, reference: Node) -> LatticeEdgeData:
    """
    Discover active-changing splits between two trees using the lattice algorithm.

    Args:
        target: Source tree for interpolation
        reference: Destination tree for interpolation

    Returns:
        LatticeEdgeData containing discovered edges and their solutions
    """
    # The lattice algorithm modifies trees during processing, so we work with copies
    target_for_lattice: Node = target.deep_copy()
    reference_for_lattice: Node = reference.deep_copy()

    lattice_edge_solutions: Dict[Partition, List[List[Partition]]] = (
        iterate_lattice_algorithm(target_for_lattice, reference_for_lattice)
    )

    lattice_edges: List[Partition] = list(lattice_edge_solutions.keys())
    active_split_data = LatticeEdgeData(lattice_edges, lattice_edge_solutions)
    active_split_data.compute_depths(target, reference)

    return active_split_data


def order_active_changing_splits_by_depth(
    edge_data: LatticeEdgeData, use_reference: bool = False, ascending: bool = True
) -> List[Partition]:
    """
    Order active-changing splits by their depth for optimal interpolation progression.
    """
    return edge_data.get_sorted_edges(use_reference=use_reference, ascending=ascending)


def generate_solution_mappings(
    solutions: Dict[Partition, List[List[Partition]]], target: Node, reference: Node
) -> Tuple[Dict[Partition, Partition], Dict[Partition, Partition]]:
    """
    Generate solution-to-atom mappings for phylogenetic analysis.
    """
    target_unique_splits: PartitionSet[Partition] = target.to_splits()
    reference_unique_splits: PartitionSet[Partition] = reference.to_splits()

    mapping_one, mapping_two = map_solutions_to_atoms(
        solutions,
        target_unique_splits,
        reference_unique_splits,
    )

    return mapping_one, mapping_two


def create_interpolation_sequence(
    target: Node,
    reference: Node,
    ordered_edges: List[Partition],
    solutions: Dict[Partition, List[List[Partition]]],
    tree_index: int,
    reference_weights: Dict[Partition, float],
) -> Tuple[
    List[Node],
    List[Partition],
    List[Optional[Partition]],
    List[Optional[Partition]],
]:
    """
    Generate the actual interpolation sequence for the given active-changing splits.
    """
    sequence_trees, failed_s_edges, s_edge_tracking, subtree_tracking = (
        orchestrate_active_split_sequence(
            target_tree=target,
            reference_tree=reference,
            reference_weights=reference_weights,
            target_active_changing_edges=ordered_edges,
            tree_index=tree_index,
            active_changing_edge_solutions=solutions,
        )
    )

    return sequence_trees, failed_s_edges, s_edge_tracking, subtree_tracking


def build_active_changing_split_interpolation_sequence(
    target: Node, reference: Node, tree_index: int
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
        target: Starting tree for interpolation
        reference: Destination tree with target topology and weights
        tree_index: Index for naming trees (e.g., 0 for T0→T1)

    Returns:
        TreePairInterpolation aggregating trees, mappings, tracking, and metrics
    """
    # Step 1: Discover active-changing splits between trees
    active_split_data: LatticeEdgeData = discover_active_changing_splits(
        target, reference
    )

    # Step 2: Order splits by depth for optimal interpolation progression
    ordered_edges = order_active_changing_splits_by_depth(
        active_split_data,
        use_reference=False,  # Use target tree depths
        ascending=True,  # Process from leaves toward root
    )

    # Step 3: Generate solution-to-atom mappings
    mappings = generate_solution_mappings(
        active_split_data.solutions, target, reference
    )

    # Step 4: Create the interpolation sequence
    reference_weights = reference.to_weighted_splits()

    trees, failed_edges, s_edge_tracking, subtree_tracking = (
        create_interpolation_sequence(
            target,
            reference,
            ordered_edges,
            active_split_data.solutions,
            tree_index,
            reference_weights,
        )
    )

    # Log any failed edges for debugging
    if failed_edges:
        logger.warning(
            f"Classical interpolation used for {len(failed_edges)} failed active-changing splits"
        )

    # Step 6: Assemble and return the final result
    mapping_one, mapping_two = mappings
    return TreePairInterpolation(
        trees=trees,
        mapping_one=mapping_one,
        mapping_two=mapping_two,
        active_changing_split_tracking=s_edge_tracking,
        subtree_tracking=subtree_tracking,
        lattice_edge_solutions=active_split_data.solutions,
    )
