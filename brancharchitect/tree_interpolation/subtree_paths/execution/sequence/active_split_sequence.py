"""Execute active-split transition sequences for one tree pair."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.types import SprMoveEvent

from ...planning import build_pivot_subtree_transition_paths
from ..pivot import execute_pivot_edge_interpolation

logger: logging.Logger = logging.getLogger(__name__)


def _offset_spr_move_events(
    events: List[SprMoveEvent], step_offset: int
) -> List[SprMoveEvent]:
    """Convert pivot-local SPR event ranges to pair-local ranges."""
    offset_events: List[SprMoveEvent] = []
    for event in events:
        step_start, step_end = event["step_range"]
        offset_events.append(
            {
                **event,
                "step_range": (step_start + step_offset, step_end + step_offset),
            }
        )
    return offset_events


def execute_active_split_transition_sequence(
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
    List[SprMoveEvent],
]:
    """
    Execute all active-changing split transitions for one source/destination pair.
    """
    interpolation_sequence: List[Node] = []
    processed_pivot_edges: List[Optional[Partition]] = []
    processed_subtree_highlights: List[Optional[List[Partition]]] = []
    spr_move_events: List[SprMoveEvent] = []

    interpolation_state: Node = source_tree.deep_copy()

    destination_subtree_paths, source_subtree_paths = (
        build_pivot_subtree_transition_paths(
            jumping_subtree_solutions, destination_tree, source_tree
        )
    )

    for current_pivot_edge in target_pivot_edges:
        current_base_tree: Node = interpolation_state

        source_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]] = (
            source_subtree_paths.get(current_pivot_edge, {})
        )
        destination_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]] = (
            destination_subtree_paths.get(current_pivot_edge, {})
        )

        source_parent_map = (
            source_parent_maps.get(current_pivot_edge) if source_parent_maps else None
        )
        dest_parent_map = (
            dest_parent_maps.get(current_pivot_edge) if dest_parent_maps else None
        )

        step_offset = len(interpolation_sequence)
        (
            step_trees,
            step_edges,
            new_state,
            step_highlights,
            step_spr_move_events,
        ) = execute_pivot_edge_interpolation(
            current_base_tree=current_base_tree,
            destination_tree=destination_tree,
            source_tree=source_tree,
            current_pivot_edge=current_pivot_edge,
            collapse_paths_for_pivot_edge=source_paths_for_pivot_edge,
            expand_paths_for_pivot_edge=destination_paths_for_pivot_edge,
            source_parent_map=source_parent_map,
            dest_parent_map=dest_parent_map,
        )

        if not step_trees:
            logger.error(
                "[active_split_sequence] Failed to generate interpolation steps for pivot edge: %s",
                current_pivot_edge,
            )
            logger.debug("Source paths: %s", source_paths_for_pivot_edge)
            logger.debug("Destination paths: %s", destination_paths_for_pivot_edge)
            raise ValueError(
                f"Interpolation failed to produce steps for pivot edge {current_pivot_edge}. "
                "The active-split transition planner could not solve the transition."
            )

        interpolation_sequence.extend(step_trees)
        processed_pivot_edges.extend(step_edges)
        processed_subtree_highlights.extend(step_highlights)
        spr_move_events.extend(
            _offset_spr_move_events(step_spr_move_events, step_offset)
        )
        interpolation_state = new_state

    return (
        interpolation_sequence,
        processed_pivot_edges,
        processed_subtree_highlights,
        spr_move_events,
    )
