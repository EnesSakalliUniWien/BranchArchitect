from __future__ import annotations

from typing import Dict, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.types import SprMoveEvent, SprPathSegment
from brancharchitect.tree_interpolation.subtree_paths.planning import PivotTransitionStep


def path_segments_with_branch_lengths(
    path: list[Partition],
    weights: Dict[Partition, float],
    side: str,
) -> list[SprPathSegment]:
    segments: list[SprPathSegment] = []
    for split in path:
        if split not in weights:
            raise KeyError(
                f"Missing {side} branch length for SPR path split {split.indices}"
            )
        segments.append(
            {
                "split": split,
                "branch_length": float(weights[split]),
            }
        )
    return segments


def build_spr_move_event(
    current_pivot_edge: Partition,
    driver_subtree: Partition,
    highlight_group: list[Partition],
    selection: PivotTransitionStep,
    source_weights: Dict[Partition, float],
    destination_weights: Dict[Partition, float],
    step_range: Tuple[int, int],
) -> SprMoveEvent:
    collapse_path = list(selection.collapse_path)
    expand_path = list(selection.expand_path)

    collapse_segments = path_segments_with_branch_lengths(
        collapse_path, source_weights, "source"
    )
    expand_segments = path_segments_with_branch_lengths(
        expand_path, destination_weights, "destination"
    )
    collapse_branch_length = sum(
        segment["branch_length"] for segment in collapse_segments
    )
    expand_branch_length = sum(segment["branch_length"] for segment in expand_segments)
    collapse_hops = len(collapse_segments)
    expand_hops = len(expand_segments)

    return {
        "pivot_edge": current_pivot_edge,
        "driver_subtree": driver_subtree,
        "highlight_group": highlight_group,
        "step_range": step_range,
        "collapse_path": collapse_segments,
        "expand_path": expand_segments,
        "collapse_hops": collapse_hops,
        "expand_hops": expand_hops,
        "total_hops": collapse_hops + expand_hops,
        "collapse_branch_length": collapse_branch_length,
        "expand_branch_length": expand_branch_length,
        "total_branch_length": collapse_branch_length + expand_branch_length,
    }


def highlight_group_for_event(
    subtree: Partition, step_highlight_tracker: list[list[Partition]]
) -> list[Partition]:
    group: list[Partition] = [subtree]
    seen: set[Partition] = {subtree}
    for frame_group in step_highlight_tracker:
        for mover in frame_group:
            if mover in seen:
                continue
            group.append(mover)
            seen.add(mover)
    return group
