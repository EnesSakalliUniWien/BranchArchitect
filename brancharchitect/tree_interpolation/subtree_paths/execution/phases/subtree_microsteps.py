from __future__ import annotations

from typing import Dict, Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

from ...planning import PivotTransitionStep
from .collapse_phase import run_collapse_phase
from .expand_phase import run_expand_phase
from .microstep_context import SelectionPaths, build_phase_highlight_groups
from .reorder_phase import run_reorder_phase
from .snap_phase import run_snap_phase
from ..frames import FrameBatch, PendingFrameBuffer


def build_subtree_interpolation_frames(
    interpolation_state: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    selection: PivotTransitionStep,
    all_mover_partitions: Optional[list[Partition]] = None,
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
    is_first_mover: bool = True,
    source_weights: Optional[Dict[Partition, float]] = None,
    destination_weights: Optional[Dict[Partition, float]] = None,
    collapse_sibling_groups: Optional[Dict[Partition, list[Partition]]] = None,
    expand_sibling_groups: Optional[Dict[Partition, list[Partition]]] = None,
) -> tuple[list[Node], list[Partition | None], Node, list[list[Partition]]]:
    """
    Build animation frames for one planner-selected driver under a pivot edge.

    The microstep flow is collapse, reorder, expand, then snap. Each phase
    mutates only private working trees or emits immutable snapshots through the
    PendingFrameBuffer.
    """
    batch = FrameBatch()
    pending_frames = PendingFrameBuffer(batch)
    paths = SelectionPaths.from_transition_step(selection)
    highlights = build_phase_highlight_groups(
        paths.subtree,
        collapse_sibling_groups,
        expand_sibling_groups,
    )

    collapsed_tree, collapsed_tree_owned = run_collapse_phase(
        interpolation_state=interpolation_state,
        destination_tree=destination_tree,
        current_pivot_edge=current_pivot_edge,
        paths=paths,
        highlights=highlights,
        pending_frames=pending_frames,
    )

    reorder_result = run_reorder_phase(
        collapsed_tree=collapsed_tree,
        collapsed_tree_owned=collapsed_tree_owned,
        destination_tree=destination_tree,
        current_pivot_edge=current_pivot_edge,
        paths=paths,
        highlights=highlights,
        pending_frames=pending_frames,
        all_mover_partitions=all_mover_partitions,
        source_parent_map=source_parent_map,
        dest_parent_map=dest_parent_map,
        is_first_mover=is_first_mover,
    )

    if not is_first_mover and not paths.has_expand_work:
        pending_frames.flush()
        if not batch.trees:
            final_tree = (
                reorder_result.tree
                if reorder_result.owned
                else reorder_result.tree.deep_copy()
            )
            return [], [], final_tree, []
        return (
            batch.trees,
            batch.edges,
            reorder_result.tree,
            batch.subtree_highlights,
        )

    if destination_weights is None:
        destination_weights = destination_tree.to_weighted_splits()

    expand_result = run_expand_phase(
        reordered_tree=reorder_result.tree,
        reordered_tree_owned=reorder_result.owned,
        has_reorder_change=reorder_result.changed,
        current_pivot_edge=current_pivot_edge,
        paths=paths,
        highlights=highlights,
        pending_frames=pending_frames,
    )

    final_tree = run_snap_phase(
        grafted_tree=expand_result.grafted_tree,
        current_pivot_edge=current_pivot_edge,
        paths=paths,
        snap_highlight_group=expand_result.snap_highlight_group,
        pending_frames=pending_frames,
        is_first_mover=is_first_mover,
        source_weights=source_weights,
        destination_weights=destination_weights,
    )

    return batch.trees, batch.edges, final_tree, batch.subtree_highlights
