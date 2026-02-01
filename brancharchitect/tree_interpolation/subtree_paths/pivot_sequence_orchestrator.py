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
from .execution.pivot_edge_interpolation_frame_builder import execute_pivot_edge_plan

logger: logging.Logger = logging.getLogger(__name__)


def find_residual_collapse_splits(
    current_pivot_edge: Partition,
    subtrees: List[Partition],
    source_only_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """
    Find residual splits that need to be collapsed but aren't on any mover's path.

    When a pivot edge like (314, 315, 316) has a singleton mover like (314,),
    there may be splits like (315, 316) that are:
    - Inside the pivot (proper subsets of pivot taxa)
    - Unique to source (not in destination)
    - Not containing any mover taxa

    These "orphaned" splits won't appear on any mover's path but still need
    to be collapsed during the pivot's processing.

    Args:
        current_pivot_edge: The pivot being processed
        subtrees: List of mover partitions for this pivot
        source_only_splits: Splits that exist only in the source tree

    Returns:
        PartitionSet of residual splits that need to be collapsed
    """
    # Collect all mover taxa for this pivot
    all_mover_taxa: set[int] = set()
    for subtree in subtrees:
        all_mover_taxa.update(subtree.indices)

    pivot_taxa = set(current_pivot_edge.indices)
    residual_splits: PartitionSet[Partition] = PartitionSet(
        encoding=source_only_splits.encoding
    )

    for s in source_only_splits:
        split_taxa = set(s.indices)
        # Must be inside pivot (proper subset)
        if split_taxa.issubset(pivot_taxa) and split_taxa != pivot_taxa:
            # Must not contain any mover taxa
            if split_taxa.isdisjoint(all_mover_taxa):
                residual_splits.add(s)

    return residual_splits


def calculate_subtree_paths(
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
    destination_tree: Node,
    source_tree: Node,
) -> tuple[
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
    Dict[Partition, Dict[Partition, PartitionSet[Partition]]],
]:
    """
    Calculates the subtree paths for both destination and source trees for each
    current pivot edge.

    Args:
        jumping_subtree_solutions: A dictionary mapping current pivot edges
            to their subtree sets.
        destination_tree: The destination tree (expand paths - splits to create).
        source_tree: The source tree (collapse paths - splits to remove).

    Returns:
        A tuple containing two dictionaries:
        - destination_subtree_paths: Paths in the destination tree, keyed by
          current pivot edge and then subtree. Used as expand paths.
        - source_subtree_paths: Paths in the source tree, keyed by
          current pivot edge and then subtree. Used as collapse paths.
    """
    destination_subtree_paths: Dict[
        Partition, Dict[Partition, PartitionSet[Partition]]
    ] = {}
    source_subtree_paths: Dict[Partition, Dict[Partition, PartitionSet[Partition]]] = {}

    # Pre-compute splits in source and destination trees
    source_splits: PartitionSet[Partition] = source_tree.to_splits()
    dest_splits: PartitionSet[Partition] = destination_tree.to_splits()

    # Identify splits that are only in source (candidates for collapse)
    source_only_splits = source_splits - dest_splits
    # Identify splits that are only in destination (candidates for expansion)
    dest_only_splits = dest_splits - source_splits

    for current_pivot_edge, subtrees in jumping_subtree_solutions.items():
        destination_subtree_paths[current_pivot_edge] = {}
        source_subtree_paths[current_pivot_edge] = {}

        # Find residual splits: orphaned splits inside the pivot that need collapsing
        residual_splits = find_residual_collapse_splits(
            current_pivot_edge, subtrees, source_only_splits
        )

        for subtree in subtrees:
            destination_node_paths: List[Node] = (
                destination_tree.find_path_between_splits(subtree, current_pivot_edge)
            )

            source_node_paths: List[Node] = source_tree.find_path_between_splits(
                subtree, current_pivot_edge
            )

            # Extract partitions from nodes
            destination_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in destination_node_paths}
            )
            source_partitions: PartitionSet[Partition] = PartitionSet(
                {node.split_indices for node in source_node_paths}
            )

            # Filter paths to only include unique splits
            # Common splits (in both trees) should use Mean interpolation and NOT be
            # collapsed or expanded/grafted.
            source_partitions = source_partitions.intersection(source_only_splits)
            destination_partitions = destination_partitions.intersection(
                dest_only_splits
            )

            # Always remove pivot edge endpoint from both paths
            destination_partitions.discard(current_pivot_edge)
            source_partitions.discard(current_pivot_edge)

            # The subtree partition itself is handled by the intersection logic:
            # - If common, it's removed from both (correct).
            # - If unique to source (shouldn't happen for mover?), it collapses.
            # - If unique to dest (new), it expands.
            # Explicit discards below are redundant but safe if strictness desired.
            source_partitions.discard(subtree)

            if subtree in source_splits:
                destination_partitions.discard(subtree)

            destination_subtree_paths[current_pivot_edge][subtree] = (
                destination_partitions
            )
            source_subtree_paths[current_pivot_edge][subtree] = source_partitions

        # Add residual splits to the first subtree's collapse path (if any)
        # These are orphaned splits inside the pivot that don't belong to any mover
        if residual_splits and source_subtree_paths[current_pivot_edge]:
            # Pick the first subtree (by bitmask order) to handle residual collapses
            first_subtree = min(
                source_subtree_paths[current_pivot_edge].keys(),
                key=lambda p: p.bitmask,
            )
            source_subtree_paths[current_pivot_edge][first_subtree] = (
                source_subtree_paths[current_pivot_edge][first_subtree]
                | residual_splits
            )

    return destination_subtree_paths, source_subtree_paths


def create_interpolation_for_active_split_sequence(
    source_tree: Node,
    destination_tree: Node,
    target_pivot_edges: List[Partition],
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
    source_parent_maps: Optional[Dict[Partition, Dict[Partition, Partition]]] = None,
    dest_parent_maps: Optional[Dict[Partition, Dict[Partition, Partition]]] = None,
    pair_index: Optional[int] = None,
) -> tuple[
    List[Node],
    List[Optional[Partition]],
    List[Optional[List[Partition]]],
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
           If no selections are produced, RAISE ERROR (strict validation).
    """
    interpolation_sequence: List[Node] = []
    processed_pivot_edge_tracking: List[Optional[Partition]] = []
    processed_subtree_tracking: List[Optional[List[Partition]]] = []

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

        # Get parent maps for this pivot edge (if available)
        source_parent_map = (
            source_parent_maps.get(current_pivot_edge) if source_parent_maps else None
        )
        dest_parent_map = (
            dest_parent_maps.get(current_pivot_edge) if dest_parent_maps else None
        )

        step_trees, step_edges, new_state, step_subtrees = execute_pivot_edge_plan(
            current_base_tree=current_base_tree,
            destination_tree=destination_tree,
            source_tree=source_tree,
            current_pivot_edge=current_pivot_edge,
            collapse_paths_for_pivot_edge=source_paths_for_pivot_edge,
            expand_paths_for_pivot_edge=destination_paths_for_pivot_edge,
            source_parent_map=source_parent_map,
            dest_parent_map=dest_parent_map,
        )

        if step_trees:
            interpolation_sequence.extend(step_trees)
            processed_pivot_edge_tracking.extend(step_edges)
            processed_subtree_tracking.extend(step_subtrees)

            interpolation_state = new_state
        else:
            # User requested strict error handling: if interpolation produces no steps
            # for a target pivot edge, it is considered a critical failure.
            # We identify exactly which pivot edge caused the issue.
            logger.error(
                f"[ORCHESTRATOR] Failed to generate interpolation steps for pivot edge: {current_pivot_edge}"
            )
            # Log the paths for debugging context
            logger.debug(f"Source Paths: {source_paths_for_pivot_edge}")
            logger.debug(f"Destination Paths: {destination_paths_for_pivot_edge}")

            raise ValueError(
                f"Interpolation failed to produce steps for pivot edge {current_pivot_edge}. "
                "This indicates the stepwise planner could not solve the transition."
            )

    return (
        interpolation_sequence,
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
