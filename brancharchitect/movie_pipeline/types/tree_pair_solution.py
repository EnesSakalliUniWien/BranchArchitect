"""Core type definitions for phylogenetic analysis."""

from typing import List, Tuple, Dict, TypedDict, Optional
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition


class TreePairSolution(TypedDict):
    """Solution data for a single tree pair."""

    # Core lattice algorithm result (CRITICAL - was missing!)
    lattice_edge_solutions: Dict[Partition, List[List[Partition]]]

    # Tree pair information
    tree_indices: Tuple[int, int]  # (start_tree_idx, end_tree_idx)

    # Mappings for atom translation
    mapping_one: Dict[Partition, Partition]  # Mapping from interpolation
    mapping_two: Dict[Partition, Partition]  # Mapping from interpolation

    # Edge sequence for tracking interpolation steps
    s_edge_sequence: List[Optional[Partition]]

    # Matching subtree sequence for each interpolation step (aligned with s_edge_sequence)
    subtree_sequence: List[Optional[Partition]]

    # S-edge distance information
    s_edge_distances: Dict[Partition, Dict[str, float]]
    """Distance information for each s-edge.

    Maps each s-edge (Partition) to a dictionary containing:
    - "target_distance": Average jump path distance from components to s-edge in target tree
    - "reference_distance": Average jump path distance from components to s-edge in reference tree
    - "total_distance": Sum of target and reference distances
    - "component_count": Number of jumping taxa (components) for this s-edge

    Example:
        {
            Partition([1, 3]): {
                "target_distance": 2.5,      # Avg path length in target tree
                "reference_distance": 1.8,   # Avg path length in reference tree
                "total_distance": 4.3,       # Combined distance
                "component_count": 4          # Number of jumping taxa
            }
        }

    Use: Analyze algorithmic complexity and phylogenetic distance of lattice solutions
    """

    # Aggregated occurrences per changing split within this pair
    split_change_events: List["SplitChangeEvent"]


class SplitChangeEvent(TypedDict):
    """
    Aggregated event for one contiguous occurrence of a changing split.

    - split: The changing split (Partition) for this event
    - step_range: Inclusive [start, end] indices, 0-based within the pair's sequence
    - subtrees: Ordered, deduplicated list of subtree Partitions observed during this event
    """

    split: Partition
    step_range: Tuple[int, int]
    subtrees: List[Partition]
