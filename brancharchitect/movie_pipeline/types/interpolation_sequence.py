from typing import Any, Dict, List, Optional, TypedDict
from brancharchitect.tree import Node

PAIR_METRIC_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "robinson_foulds": {
        "topology": "unrooted_internal_bipartitions",
        "normalization": "symmetric_difference_over_split_count_sum",
        "scope": "adjacent_processed_input_trees",
    },
    "weighted_robinson_foulds": {
        "topology": "rooted_clades",
        "includes_branch_lengths": True,
        "includes_terminal_and_root_splits": True,
        "scope": "adjacent_processed_input_trees",
    },
}


class InterpolationResult(TypedDict):
    """Primary backend result consumed by the frontend stream builder."""

    interpolated_trees: List[Node]
    frames: List[Dict[str, Any]]
    pairs: List[Dict[str, Any]]
    temporal_events: List[Dict[str, Any]]
    pair_metrics: Dict[str, Any]
    processing_time: float
    subtree_highlight_tracking: List[Optional[List[List[int]]]]


def create_single_tree_result(
    trees: List[Node],
) -> InterpolationResult:
    """Create an InterpolationResult for a single tree case."""

    return InterpolationResult(
        interpolated_trees=trees,
        frames=[
            {
                "frame_index": 0,
                "frame_type": "input_tree",
                "state_semantics": "processed_input_tree",
                "is_observed_input": True,
                "input_tree_index": 0,
                "pair_id": None,
                "pair_ordinal": None,
                "local_step_index": None,
                "source_frame_index": None,
                "target_frame_index": None,
            }
        ],
        pairs=[],
        temporal_events=[],
        pair_metrics={"rows": [], "semantics": PAIR_METRIC_SEMANTICS},
        processing_time=0.0,
        subtree_highlight_tracking=[None],  # Single tree has no interpolation highlight
    )


def create_empty_result() -> InterpolationResult:
    """Create an empty InterpolationResult for edge cases like empty tree lists."""
    return InterpolationResult(
        interpolated_trees=[],
        frames=[],
        pairs=[],
        temporal_events=[],
        pair_metrics={"rows": [], "semantics": PAIR_METRIC_SEMANTICS},
        processing_time=0.0,
        subtree_highlight_tracking=[],
    )
