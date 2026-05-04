"""Core type definitions for phylogenetic analysis."""

from typing import List, Dict, NotRequired, TypedDict
from brancharchitect.elements.partition import Partition


class TreePairSolution(TypedDict):
    """Solution data for a single tree pair."""

    # Core jumping taxa algorithm result - solutions for subtree rearrangements
    jumping_subtree_solutions: Dict[Partition, List[Partition]]

    # Mappings for atom translation - where movers come from and go to
    solution_to_destination_map: Dict[
        Partition, Dict[Partition, Partition]
    ]  # Mapping from solution to destination tree atoms, grouped by pivot edge
    solution_to_source_map: Dict[
        Partition, Dict[Partition, Partition]
    ]  # Mapping from solution to source tree atoms, grouped by pivot edge

    # Aggregated occurrences per changing split within this pair
    split_change_events: List["SplitChangeEvent"]

    # Per-SPR movement context, including path hops and branch lengths
    spr_move_events: NotRequired[List["SprMoveEvent"]]

    # Global indices of the source and destination trees in the complete interpolated sequence
    source_tree_global_index: int
    """Global index of the source tree this pair interpolates FROM."""

    destination_tree_global_index: int
    """Global index of the destination tree this pair interpolates TO."""

    interpolation_start_global_index: int
    """Global index where interpolated trees for this pair begin (first interpolated tree)."""


class SplitChangeEvent(TypedDict):
    """
    Aggregated event for one contiguous occurrence of a changing split.

    - split: The changing split (Partition) for this event
    - step_range: Inclusive [start, end] indices, 0-based within the pair's sequence
    - source_tree_global_index: Global index of the source tree for this event
    - destination_tree_global_index: Global index of the destination tree for this event
    This version does not track subtrees at the frontend anymore.
    """

    split: Partition
    step_range: tuple[int, int]
    source_tree_global_index: int
    destination_tree_global_index: int


class SprPathSegment(TypedDict):
    """One split traversed by an SPR collapse or expand path."""

    split: Partition
    branch_length: float


class SprMoveEvent(TypedDict):
    """Path summary for one SPR mover within a tree-pair interpolation."""

    pivot_edge: Partition
    moving_subtree: Partition
    step_range: tuple[int, int]
    collapse_path: List[SprPathSegment]
    expand_path: List[SprPathSegment]
    collapse_hops: int
    expand_hops: int
    total_hops: int
    collapse_branch_length: float
    expand_branch_length: float
    total_branch_length: float
