"""Finalize branch weights for one mover microstep."""

from __future__ import annotations

from typing import Dict, Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.topology_ops.weights import (
    finalize_branch_weights,
)

from .microstep_context import SelectionPaths
from ..frames import PendingFrameBuffer


def run_snap_phase(
    grafted_tree: Node,
    current_pivot_edge: Partition,
    paths: SelectionPaths,
    snap_highlight_group: list[Partition],
    pending_frames: PendingFrameBuffer,
    is_first_mover: bool,
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> Node:
    """Snap branch lengths to the source/destination weights for this step."""
    finalize_branch_weights(
        tree=grafted_tree,
        current_pivot_edge=current_pivot_edge,
        expand_path=paths.expand_paths,
        is_first_mover=is_first_mover,
        source_weights=source_weights,
        destination_weights=destination_weights,
    )

    pending_frames.batch.append(
        grafted_tree.deep_copy(build_split_index=False),
        current_pivot_edge,
        snap_highlight_group,
    )

    return grafted_tree
