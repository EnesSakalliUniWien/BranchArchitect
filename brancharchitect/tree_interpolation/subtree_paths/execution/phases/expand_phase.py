"""Expand destination-only splits for one mover microstep."""

from __future__ import annotations

from dataclasses import dataclass

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.topology_ops.expand import (
    create_subtree_grafted_tree,
)

from .microstep_context import PhaseHighlightGroups, SelectionPaths
from ..frames import PendingFrameBuffer
from ..layout import align_to_source_order
from ..layout.mover_ordering import (
    expand_paths_require_leaf_order_change,
    taxa_for_partitions,
)


@dataclass(frozen=True, slots=True)
class ExpandPhaseResult:
    grafted_tree: Node
    snap_highlight_group: list[Partition]


def run_expand_phase(
    reordered_tree: Node,
    reordered_tree_owned: bool,
    has_reorder_change: bool,
    current_pivot_edge: Partition,
    paths: SelectionPaths,
    highlights: PhaseHighlightGroups,
    pending_frames: PendingFrameBuffer,
) -> ExpandPhaseResult:
    """Graft destination-only splits into the pivot context."""
    if not paths.has_expand_work:
        grafted_tree = reordered_tree.deep_copy()
        pending_frames.flush()
        return ExpandPhaseResult(
            grafted_tree=grafted_tree,
            snap_highlight_group=highlights.reorder,
        )

    reordered_order = list(reordered_tree.get_current_order())
    expand_may_reorder = expand_paths_require_leaf_order_change(
        paths.expand_paths,
        reordered_order,
    )

    pending_frames.snapshot_if_pending_tree_is(reordered_tree)

    if not reordered_tree_owned:
        reordered_tree = reordered_tree.deep_copy()
        reordered_tree_owned = True

    pre_graft_tree = (
        reordered_tree.deep_copy(build_split_index=False)
        if expand_may_reorder
        else None
    )

    grafted_zero_weights: Node = create_subtree_grafted_tree(
        base_tree=reordered_tree,
        ref_path_to_build=paths.expand_paths,
        copy=False,
    )

    grafted_zero_weights.reorder_taxa(reordered_order)

    align_to_source_order(
        grafted_zero_weights,
        source_order=reordered_order,
        moving_taxa=taxa_for_partitions(highlights.expand),
    )
    grafted_order = list(grafted_zero_weights.get_current_order())

    if grafted_order != reordered_order:
        if pre_graft_tree is None:
            raise RuntimeError(
                "Expand changed leaf order although all expand splits were "
                "contiguous before grafting"
            )
        pre_graft_tree.reorder_taxa(grafted_order)
        if has_reorder_change and pending_frames.has_pending:
            pending_frames.replace(
                pre_graft_tree,
                current_pivot_edge,
                highlights.reorder,
            )
        else:
            pending_frames.set(
                pre_graft_tree,
                current_pivot_edge,
                highlights.reorder,
            )

    pending_frames.flush()
    pending_frames.batch.append(
        grafted_zero_weights.deep_copy(build_split_index=False),
        current_pivot_edge,
        highlights.expand,
    )

    return ExpandPhaseResult(
        grafted_tree=grafted_zero_weights,
        snap_highlight_group=highlights.expand,
    )
