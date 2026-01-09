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
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    topological_sort_edges,
)
from brancharchitect.jumping_taxa.lattice.mapping.minimum_cover_mappings import (
    map_solution_elements_via_parent,
)

# Final topology check: ensure last interpolated tree matches destination
from brancharchitect.tree_interpolation.subtree_paths.pivot_sequence_orchestrator import (
    assert_final_topology_matches,
)
from brancharchitect.tree_interpolation.types import (
    TreePairInterpolation,
)
# distance_metrics import removed (metrics no longer used here)

logger: logging.Logger = logging.getLogger(__name__)

__all__: List[str] = [
    "process_tree_pair_interpolation",
]


def process_tree_pair_interpolation(
    source_tree: Node,
    destination_tree: Node,
    precomputed_solutions: Optional[Dict[Partition, List[Partition]]] = None,
    pair_index: Optional[int] = None,
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
    # Step 1: Discover active-changing splits via lattice algorithm
    # The caller (SequentialInterpolationBuilder) already passes deep-copied trees
    # for each pair, so we can call the lattice algorithm directly without
    # performing additional copies here.

    if precomputed_solutions is not None:
        jumping_subtree_solutions: Dict[Partition, List[Partition]] = (
            precomputed_solutions
        )
    else:
        jumping_subtree_solutions, _ = LatticeSolver(
            source_tree, destination_tree
        ).solve_iteratively()

    # Sort pivot edges topologically (subsets before supersets)
    # This ensures correct processing order for interpolation
    pivot_edges: List[Partition] = list(jumping_subtree_solutions.keys())
    ordered_edges: List[Partition] = topological_sort_edges(pivot_edges, source_tree)

    # Compute MRCA parent maps for all movers (V2 MRCA-aware reordering)
    source_parent_maps, dest_parent_maps = map_solution_elements_via_parent(
        jumping_subtree_solutions, source_tree, destination_tree
    )

    (
        sequence_trees,
        current_pivot_edge_tracking,
        current_subtree_tracking,
    ) = create_interpolation_for_active_split_sequence(
        source_tree=source_tree,
        destination_tree=destination_tree,
        target_pivot_edges=ordered_edges,
        jumping_subtree_solutions=jumping_subtree_solutions,
        source_parent_maps=source_parent_maps,
        dest_parent_maps=dest_parent_maps,
        pair_index=pair_index,
    )

    if sequence_trees:
        # User Request: Throw error on mismatch instead of fallback
        assert_final_topology_matches(sequence_trees[-1], destination_tree, logger)

    # For identical trees (no active edges), ensure destination tree has same ordering as source
    if not ordered_edges:
        logger.debug(
            "No active-changing edges found - trees are identical. Ensuring consistent ordering."
        )
        # Copy the leaf ordering from source to destination tree
        source_order = source_tree.get_current_order()
        destination_tree.reorder_taxa(list(source_order))

    return TreePairInterpolation(
        trees=sequence_trees,
        current_pivot_edge_tracking=current_pivot_edge_tracking,
        jumping_subtree_solutions=jumping_subtree_solutions,
        current_subtree_tracking=current_subtree_tracking,
    )
