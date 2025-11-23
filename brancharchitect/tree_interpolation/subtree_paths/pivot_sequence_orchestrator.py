"""Sequence orchestrator for subtree-path interpolation.

Coordinates the high-level interpolation sequence for each pivot edge (current_pivot_edge)
split by:
- Calculating per-split subtree paths
- Planning collapse/expand steps per subtree
- Executing the 5 microsteps and aggregating results

Terminology:
    - pivot_edge / current_pivot_edge: The split being processed (preferred term)
    - active-changing split: Formal documentation term (same concept)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .paths import calculate_subtree_paths
from .execution.step_executor import apply_stepwise_plan_for_edge

logger = logging.getLogger(__name__)


def create_interpolation_for_active_split_sequence(
    source_tree: Node,
    destination_tree: Node,
    target_pivot_edges: List[Partition],
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
) -> tuple[
    List[Node],
    List[Partition],
    List[Optional[Partition]],
]:
    """
    Create an interpolation sequence from source to destination tree for pivot edges (active-changing splits).

    Strategy (modular and stepwise):
      1) Calculate subtree paths once for destination and source.
      2) For each current_pivot_edge on the current interpolation state:
         - If the current_pivot_edge is missing in either tree, run classical fallback.
         - Else, try stepwise plan:
             a) Iterate selections (individual/whole per rule).
             b) Apply micro-steps per selection.
           If no selections are produced, run the simple 5-step fallback for that current_pivot_edge.
    """
    interpolation_sequence: List[Node] = []
    failed_pivot_edges: List[Partition] = []
    processed_pivot_edge_tracking: List[Optional[Partition]] = []
    # subtree tracking removed

    interpolation_state: Node = source_tree.deep_copy()

    # Precompute subtree paths in destination/source
    destination_subtree_paths, source_subtree_paths = calculate_subtree_paths(
        jumping_subtree_solutions, destination_tree, source_tree
    )

    current_base_tree = interpolation_state.deep_copy()

    for current_pivot_edge in target_pivot_edges:
        current_base_tree: Node = interpolation_state.deep_copy()

        current_base_tree.initialize_split_indices(current_base_tree.taxa_encoding)

        # Paths for this current_pivot_edge (kept separate for destination/source)
        # Keep as PartitionSet[Partition] - no conversion needed
        source_paths_for_pivot_edge = source_subtree_paths.get(current_pivot_edge, {})

        destination_paths_for_pivot_edge = destination_subtree_paths.get(
            current_pivot_edge, {}
        )

        # Guard: verify the pivot edge exists in both trees before planning
        src_node, dst_node = _find_and_validate_pivot_nodes(
            current_base_tree, destination_tree, current_pivot_edge
        )

        step_trees, step_edges, new_state = apply_stepwise_plan_for_edge(
            current_base_tree=current_base_tree,
            destination_tree=destination_tree,
            current_pivot_edge=current_pivot_edge,
            expand_paths_for_pivot_edge=destination_paths_for_pivot_edge,
            collapse_paths_for_pivot_edge=source_paths_for_pivot_edge,
        )

        if step_trees:
            interpolation_sequence.extend(step_trees)
            processed_pivot_edge_tracking.extend(step_edges)

            interpolation_state = new_state

    return (
        interpolation_sequence,
        failed_pivot_edges,
        processed_pivot_edge_tracking,
    )


def assert_final_topology_matches(
    final_state: Node, destination_tree: Node, logger: logging.Logger
) -> None:
    """
    Helper to verify final interpolation state matches destination topology.

    Logs an error with missing/extra splits if a mismatch is detected.
    """
    final_splits = final_state.to_splits()
    dest_splits = destination_tree.to_splits()

    if final_splits != dest_splits:
        missing = dest_splits - final_splits
        extra = final_splits - dest_splits
        logger.error(
            "[ORCHESTRATOR] Final topology mismatch: expected %s splits, got %s",
            len(dest_splits),
            len(final_splits),
        )
        if missing:
            logger.error(
                "[ORCHESTRATOR]   Missing splits: %s",
                {tuple(s.indices) for s in missing},
            )
        if extra:
            logger.error(
                "[ORCHESTRATOR]   Extra splits: %s",
                {tuple(s.indices) for s in extra},
            )


def _ensure_pivot_present(
    current_pivot_edge: Partition,
    src_node: Optional[Node],
    dst_node: Optional[Node],
) -> None:
    """
    Validate that the pivot edge exists in both source and destination trees.

    Raises a ValueError with a descriptive message if the edge is missing.
    """
    if src_node is None or dst_node is None:
        missing_in: list[str] = []
        if src_node is None:
            missing_in.append("source (current interpolation state)")
        if dst_node is None:
            missing_in.append("destination")

        raise ValueError(
            f"Pivot edge {current_pivot_edge.bipartition()} "
            f"missing in {' and '.join(missing_in)} tree. "
            f"This edge was identified by the lattice algorithm but doesn't exist in both trees. "
            f"See debug logs above for edge comparison."
        )


def _find_and_validate_pivot_nodes(
    current_base_tree: Node, destination_tree: Node, current_pivot_edge: Partition
) -> tuple[Optional[Node], Optional[Node]]:
    """Locate pivot nodes in both trees and validate presence."""
    src_node = current_base_tree.find_node_by_split(current_pivot_edge)
    dst_node = destination_tree.find_node_by_split(current_pivot_edge)
    _ensure_pivot_present(current_pivot_edge, src_node, dst_node)
    return src_node, dst_node
