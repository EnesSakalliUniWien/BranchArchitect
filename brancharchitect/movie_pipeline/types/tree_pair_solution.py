"""Core type definitions for phylogenetic analysis."""

from typing import List, Dict, TypedDict, Optional
from brancharchitect.elements.partition import Partition


class TreePairSolution(TypedDict):
    """Solution data for a single tree pair."""

    # Core jumping taxa algorithm result - solutions for subtree rearrangements
    jumping_subtree_solutions: Dict[Partition, List[Partition]]

    # Mappings for atom translation (actual keys used in implementation)
    solution_to_target_map: Dict[
        Partition, Dict[Partition, Partition]
    ]  # Mapping from solution to target tree atoms, grouped by pivot edge
    solution_to_reference_map: Dict[
        Partition, Dict[Partition, Partition]
    ]  # Mapping from solution to reference tree atoms, grouped by pivot edge
    
    # Backward compatibility aliases (if needed for existing code)
    # mapping_one: Dict[Partition, Partition]  # Alias for solution_to_target_map
    # mapping_two: Dict[Partition, Partition]  # Alias for solution_to_reference_map

    # Ancestors of the changing splits for each interpolation step
    ancestor_of_changing_splits: List[Optional[Partition]]

    # Aggregated occurrences per changing split within this pair
    split_change_events: List["SplitChangeEvent"]

    # Global indices of the source and target trees in the complete interpolated sequence
    source_tree_global_index: int
    """Global index of the source tree this pair interpolates FROM."""
    
    target_tree_global_index: int
    """Global index of the target tree this pair interpolates TO."""
    
    interpolation_start_global_index: int
    """Global index where interpolated trees for this pair begin (first interpolated tree)."""


class SplitChangeEvent(TypedDict):
    """
    Aggregated event for one contiguous occurrence of a changing split.

    - split: The changing split (Partition) for this event
    - step_range: Inclusive [start, end] indices, 0-based within the pair's sequence
    - source_tree_global_index: Global index of the source tree for this event
    - target_tree_global_index: Global index of the target tree for this event
    This version does not track subtrees at the frontend anymore.
    """

    split: Partition
    step_range: tuple[int, int]
    source_tree_global_index: int
    target_tree_global_index: int
