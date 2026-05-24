"""
Builds the frontend-specific data structures from the backend processing result.

This module transforms the raw output from TreeInterpolationPipeline into
the exact format required by the frontend UI.

Key Responsibilities:
- Serialize tree objects for chunked streaming.
- Assemble the normalized metadata payload sent before streamed tree chunks.
"""

from __future__ import annotations

from typing import Any, Dict

from brancharchitect.io import serialize_tree_list_to_json
from brancharchitect.movie_pipeline.types import (
    InterpolationResult,
    PAIR_METRIC_SEMANTICS,
)
from webapp.services.trees.movie_data import MovieData

# =============================================================================
# Main Entry Points
# =============================================================================


def build_movie_data_from_result(
    result: InterpolationResult,
    filename: str,
    msa_data: Dict[str, Any],
) -> MovieData:
    """
    Create a MovieData instance from the backend's InterpolationResult.

    This is the main entry point for this module. It orchestrates the
    transformation of backend data into a structured MovieData object.
    """
    interpolated_trees = result["interpolated_trees"]
    serialized_trees = serialize_tree_list_to_json(interpolated_trees)

    return MovieData(
        interpolated_trees=serialized_trees,
        frames=result["frames"],
        pairs=result["pairs"],
        temporal_events=result["temporal_events"],
        pair_metrics=result["pair_metrics"],
        subtree_highlight_tracking=result["subtree_highlight_tracking"],
        file_name=filename,
        window_size=msa_data["inferred_window_size"],
        window_step_size=msa_data["inferred_step_size"],
        msa_dict=msa_data["msa_dict"],
    )


def assemble_frontend_metadata(movie_data: MovieData) -> Dict[str, Any]:
    """
    Create the movie metadata payload sent before tree chunks.
    """
    return {
        "frames": movie_data.frames,
        "pairs": movie_data.pairs,
        "temporal_events": movie_data.temporal_events,
        "subtree_highlight_tracking": movie_data.subtree_highlight_tracking,
        "pair_metrics": movie_data.pair_metrics,
        "msa": {
            "sequences": movie_data.msa_dict,
            "window_size": movie_data.window_size,
            "step_size": movie_data.window_step_size,
        },
        "file_name": movie_data.file_name,
    }


def create_empty_movie_data(filename: str) -> MovieData:
    """Create empty MovieData for failed processing scenarios."""
    return MovieData(
        interpolated_trees=[],
        frames=[],
        pairs=[],
        temporal_events=[],
        pair_metrics={"rows": [], "semantics": PAIR_METRIC_SEMANTICS},
        subtree_highlight_tracking=[],
        file_name=filename,
        window_size=1,
        window_step_size=1,
        msa_dict=None,
    )
