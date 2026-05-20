"""Per-pair interpolation movement type definitions."""

from typing import List, TypedDict
from brancharchitect.elements.partition import Partition


class AttachmentEdges(TypedDict):
    """Source and destination attachment edges for one moved subtree."""

    source: Partition
    destination: Partition


class SprPathSegment(TypedDict):
    """One split traversed by an SPR collapse or expand path."""

    split: Partition
    branch_length: float


class SprMoveEvent(TypedDict):
    """Path summary for one SPR mover within a tree-pair interpolation.

    driver_subtree is the planner-selected subtree that physically moves for
    this SPR event. highlight_group is the active mover highlight set used by
    per-frame current_subtree_highlights; it may include explicit sibling mover
    groups, but not passive context subtrees.
    """

    pivot_edge: Partition
    driver_subtree: Partition
    highlight_group: List[Partition]
    step_range: tuple[int, int]
    collapse_path: List[SprPathSegment]
    expand_path: List[SprPathSegment]
    collapse_hops: int
    expand_hops: int
    total_hops: int
    collapse_branch_length: float
    expand_branch_length: float
    total_branch_length: float
