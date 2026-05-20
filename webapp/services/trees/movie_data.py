"""
Movie data class for serializing backend responses to frontend format.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MovieData:
    """
    Pure data class that holds the final, frontend-ready tree processing results.

    This class is a pure data container with no dependencies on other modules.
    All construction and conversion logic is in frontend_builder.py.
    """

    # Core tree data
    interpolated_trees: List[Dict[str, Any]]
    frames: List[Dict[str, Any]]
    pairs: List[Dict[str, Any]]
    temporal_events: List[Dict[str, Any]]
    pair_metrics: Dict[str, Any]

    # Visualization data
    pivot_edge_tracking: List[Optional[List[int]]]
    # Per-frame visual/highlight groups, not SPR mover ownership.
    subtree_highlight_tracking: List[Optional[List[List[int]]]]

    # File and processing metadata
    file_name: str
    window_size: int
    window_step_size: int

    # MSA data
    msa_dict: Optional[Dict[str, str]]
