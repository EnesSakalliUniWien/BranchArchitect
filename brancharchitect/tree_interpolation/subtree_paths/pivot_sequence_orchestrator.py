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
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .paths import calculate_subtree_paths
from .execution.step_executor import apply_stepwise_plan_for_edge

logger: logging.Logger = logging.getLogger(__name__)


def create_interpolation_for_active_split_sequence(
    source_tree: Node,
    destination_tree: Node,
    target_pivot_edges: List[Partition],
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
    pair_index: Optional[int] = None,
) -> tuple[
    List[Node],
    List[Partition],
    List[Optional[Partition]],
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
    processed_subtree_tracking: List[Optional[Partition]] = []

    interpolation_state: Node = source_tree.deep_copy()

    # Precompute subtree paths in destination/source
    destination_subtree_paths, source_subtree_paths = calculate_subtree_paths(
        jumping_subtree_solutions, destination_tree, source_tree
    )

    for current_pivot_edge in target_pivot_edges:
        current_base_tree: Node = interpolation_state.deep_copy()

        # Ensure split indices are initialized for the copied tree
        # deep_copy preserves split_indices per node, but _split_index cache needs rebuild
        current_base_tree.build_split_index()

        # Paths for this current_pivot_edge (kept separate for destination/source)
        # Keep as PartitionSet[Partition] - no conversion needed
        source_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]] = (
            source_subtree_paths.get(current_pivot_edge, {})
        )

        destination_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]] = (
            destination_subtree_paths.get(current_pivot_edge, {})
        )

        # Guard: verify the pivot edge exists in both trees before planning
        _ = _find_and_validate_pivot_nodes(
            current_base_tree, destination_tree, current_pivot_edge
        )

        step_trees, step_edges, new_state, step_subtrees = apply_stepwise_plan_for_edge(
            current_base_tree=current_base_tree,
            destination_tree=destination_tree,
            current_pivot_edge=current_pivot_edge,
            expand_paths_for_pivot_edge=destination_paths_for_pivot_edge,
            collapse_paths_for_pivot_edge=source_paths_for_pivot_edge,
        )

        if step_trees:
            interpolation_sequence.extend(step_trees)
            processed_pivot_edge_tracking.extend(step_edges)
            processed_subtree_tracking.extend(step_subtrees)

            interpolation_state = new_state

    return (
        interpolation_sequence,
        failed_pivot_edges,
        processed_pivot_edge_tracking,
        processed_subtree_tracking,
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
        msg = (
            "[ORCHESTRATOR] Final topology mismatch: "
            f"expected {len(dest_splits)} splits, got {len(final_splits)}. "
            f"Missing splits: { {tuple(s.indices) for s in missing} } "
            f"Extra splits: { {tuple(s.indices) for s in extra} }"
        )
        logger.error(msg)
        raise ValueError(msg)


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
