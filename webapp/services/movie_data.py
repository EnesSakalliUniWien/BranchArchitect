"""Movie data class for serializing backend responses to frontend format."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from brancharchitect.movie_pipeline.types import (
    TreeMetadata as TreeMetadataType,
    TreePairSolution,
)


@dataclass
class MovieData:
    """
    Pure data class that holds the final, frontend-ready tree processing results.

    This class is now a pure data container with no dependencies on other modules.
    All construction and conversion logic has been moved to frontend_data_builder.py
    to eliminate circular imports.
    """

    # Core tree data
    interpolated_trees: List[Dict[str, Any]]
    tree_metadata: List[TreeMetadataType]

    # Distance metrics
    rfd_list: List[float]
    weighted_robinson_foulds_distance_list: List[float]

    # Visualization data
    sorted_leaves: List[str]
    tree_pair_solutions: Dict[str, TreePairSolution]
    split_change_tracking: List[Optional[List[int]]]

    # File and processing metadata
    file_name: str
    window_size: int
    window_step_size: int

    # MSA data
    msa_dict: Optional[Dict[str, str]]
    alignment_length: Optional[int]
    windows_are_overlapping: bool

    # Processing metadata
    original_tree_count: int
    interpolated_tree_count: int
    rooting_enabled: bool
    pair_interpolation_ranges: List[List[int]]
