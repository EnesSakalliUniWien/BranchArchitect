"""Core type definitions for phylogenetic analysis."""

from typing import List, Dict, TypedDict, Optional
from brancharchitect.elements.partition import Partition


class TreePairSolution(TypedDict):
    """Solution data for a single tree pair."""

    # Core lattice algorithm result (CRITICAL - was missing!)
    lattice_edge_solutions: Dict[Partition, List[List[Partition]]]

    # Mappings for atom translation
    mapping_one: Dict[Partition, Partition]  # Mapping from interpolation
    mapping_two: Dict[Partition, Partition]  # Mapping from interpolation

    # Ancestors of the changing splits for each interpolation step
    ancestor_of_changing_splits: List[Optional[Partition]]

    # Aggregated occurrences per changing split within this pair
    split_change_events: List["SplitChangeEvent"]


class SplitChangeEvent(TypedDict):
    """
    Aggregated event for one contiguous occurrence of a changing split.

    - split: The changing split (Partition) for this event
    - step_range: Inclusive [start, end] indices, 0-based within the pair's sequence
    This version does not track subtrees at the frontend anymore.
    """

    split: Partition
    step_range: tuple[int, int]
