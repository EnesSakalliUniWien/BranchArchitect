"""Collapse source-only splits for one mover microstep."""

from __future__ import annotations

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.topology_ops.weights import (
    apply_zero_branch_lengths,
)

from .microstep_context import PhaseHighlightGroups, SelectionPaths
from ..frames import PendingFrameBuffer


def run_collapse_phase(
    interpolation_state: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    paths: SelectionPaths,
    highlights: PhaseHighlightGroups,
    pending_frames: PendingFrameBuffer,
) -> tuple[Node, bool]:
    """Zero and collapse source-only splits for the current mover."""
    if not paths.has_collapse_work:
        return interpolation_state, False

    zeroed_tree: Node = interpolation_state
    zeroed_order = list(zeroed_tree.get_current_order())

    apply_zero_branch_lengths(zeroed_tree, PartitionSet(set(paths.collapse_paths)))

    collapsed_tree: Node = create_collapsed_consensus_tree(
        zeroed_tree,
        current_pivot_edge,
        copy=True,
        destination_tree=destination_tree,
    )
    collapsed_tree.reorder_taxa(zeroed_order)

    pending_frames.set(
        zeroed_tree,
        current_pivot_edge,
        highlights.collapse,
    )
    pending_frames.set(
        collapsed_tree,
        current_pivot_edge,
        highlights.collapse,
    )

    return collapsed_tree, True
