"""Reorder the current pivot context toward destination layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

from .microstep_context import PhaseHighlightGroups, SelectionPaths
from ..frames import PendingFrameBuffer
from ..layout import reorder_tree_toward_destination


@dataclass(frozen=True, slots=True)
class ReorderPhaseResult:
    tree: Node
    owned: bool
    changed: bool


def run_reorder_phase(
    collapsed_tree: Node,
    collapsed_tree_owned: bool,
    destination_tree: Node,
    current_pivot_edge: Partition,
    paths: SelectionPaths,
    highlights: PhaseHighlightGroups,
    pending_frames: PendingFrameBuffer,
    all_mover_partitions: Optional[list[Partition]],
    source_parent_map: Optional[Dict[Partition, Partition]],
    dest_parent_map: Optional[Dict[Partition, Partition]],
    is_first_mover: bool,
) -> ReorderPhaseResult:
    """Move the current mover block toward its destination order."""
    pre_reorder_order = tuple(collapsed_tree.get_current_order())
    pending_aliases_collapsed_tree = pending_frames.is_pending_tree(collapsed_tree)
    reordered_tree: Node = reorder_tree_toward_destination(
        source_tree=collapsed_tree,
        destination_tree=destination_tree,
        current_pivot_edge=current_pivot_edge,
        moving_subtree_partition=paths.subtree,
        source_parent_map=source_parent_map,
        dest_parent_map=dest_parent_map,
        unstable_mover_partitions=all_mover_partitions,
        copy=(not collapsed_tree_owned) or pending_aliases_collapsed_tree,
    )

    has_reorder_change = tuple(reordered_tree.get_current_order()) != pre_reorder_order
    reordered_tree_owned = has_reorder_change or collapsed_tree_owned

    if not has_reorder_change:
        return ReorderPhaseResult(
            tree=collapsed_tree,
            owned=reordered_tree_owned,
            changed=False,
        )

    if not paths.has_collapse_work:
        pre_reorder_frame = (
            collapsed_tree
            if collapsed_tree_owned
            else collapsed_tree.deep_copy(build_split_index=False)
        )
        pending_frames.set(
            pre_reorder_frame,
            current_pivot_edge,
            highlights.reorder,
        )

    reorder_frame_tree = (
        reordered_tree.deep_copy(build_split_index=False)
        if not is_first_mover and not paths.has_expand_work
        else reordered_tree
    )
    pending_frames.set(
        reorder_frame_tree,
        current_pivot_edge,
        highlights.reorder,
    )

    return ReorderPhaseResult(
        tree=reordered_tree,
        owned=reordered_tree_owned,
        changed=True,
    )
