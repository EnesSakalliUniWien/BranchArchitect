"""
Movie data class for serializing backend responses to frontend format.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brancharchitect.movie_pipeline.types import (
    TreeMetadata as TreeMetadataType,
    TreePairSolution,
)


@dataclass
class MovieData:
    """
    Pure data class that holds the final, frontend-ready tree processing results.

    This class is a pure data container with no dependencies on other modules.
    All construction and conversion logic is in frontend_builder.py.
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
    pivot_edge_tracking: List[Optional[List[int]]]
    subtree_tracking: List[Optional[List[int]]]

    # File and processing metadata
    file_name: str
    window_size: int
    window_step_size: int

    # MSA data
    msa_dict: Optional[Dict[str, str]]
    pair_interpolation_ranges: List[List[int]]
