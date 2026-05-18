"""
Builds the frontend-specific data structures from the backend processing result.

This module transforms the raw output from TreeInterpolationPipeline into
the exact format required by the frontend UI.

Key Responsibilities:
- Serialize rich Python objects (like Partitions) into simple JSON types.
- Derive UI-specific data structures like pivot_edge_tracking and split_change_timeline.
- Assemble the metadata payload sent before streamed tree chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from brancharchitect.io import serialize_tree_list_to_json
from brancharchitect.movie_pipeline.types import (
    InterpolationResult,
    TreeMetadata as TreeMetadataType,
    TreePairSolution,
)
from brancharchitect.tree_interpolation.types import SprMoveEvent, SprPathSegment

from webapp.services.serialization import (
    serialize_partition_dict_to_indices,
    serialize_partition_to_indices,
)
from webapp.services.trees.movie_data import MovieData

# =============================================================================
# Main Entry Points
# =============================================================================


def build_movie_data_from_result(
    result: InterpolationResult,
    filename: str,
    msa_data: Dict[str, Any],
    sorted_leaves: List[str],
) -> MovieData:
    """
    Create a MovieData instance from the backend's InterpolationResult.

    This is the main entry point for this module. It orchestrates the
    transformation of backend data into a structured MovieData object.
    """
    interpolated_trees = result["interpolated_trees"]
    serialized_trees = serialize_tree_list_to_json(interpolated_trees)
    tree_metadata = _process_tree_metadata(result["tree_metadata"])

    events_by_pair = _extract_split_change_events_from_solutions(
        result["tree_pair_solutions"]
    )
    pivot_edge_tracking = _derive_pivot_edge_tracking_from_events(
        tree_metadata, events_by_pair
    )

    return MovieData(
        interpolated_trees=serialized_trees,
        tree_metadata=tree_metadata,
        rfd_list=result.get("rfd_list", []),
        weighted_robinson_foulds_distance_list=result.get("wrfd_list", []),
        sorted_leaves=sorted_leaves,
        tree_pair_solutions=result["tree_pair_solutions"],
        pivot_edge_tracking=pivot_edge_tracking,
        subtree_highlight_tracking=result["subtree_highlight_tracking"],
        file_name=filename,
        window_size=msa_data.get("inferred_window_size", 1),
        window_step_size=msa_data.get("inferred_step_size", 1),
        msa_dict=msa_data.get("msa_dict"),
        pair_interpolation_ranges=result.get("pair_interpolation_ranges", []),
    )


def assemble_frontend_metadata(movie_data: MovieData) -> Dict[str, Any]:
    """
    Create the movie metadata payload sent before tree chunks.
    """
    timeline = _build_split_change_timeline(
        movie_data.tree_metadata,
        movie_data.tree_pair_solutions,
    )

    return {
        "tree_metadata": movie_data.tree_metadata,
        "tree_pair_solutions": _serialize_tree_pair_solutions(
            movie_data.tree_pair_solutions
        ),
        "split_change_timeline": timeline,
        "sorted_leaves": movie_data.sorted_leaves,
        "pivot_edge_tracking": movie_data.pivot_edge_tracking,
        "subtree_highlight_tracking": movie_data.subtree_highlight_tracking,
        "pair_interpolation_ranges": movie_data.pair_interpolation_ranges,
        "msa": {
            "sequences": movie_data.msa_dict,
            "window_size": movie_data.window_size,
            "step_size": movie_data.window_step_size,
        },
        "file_name": movie_data.file_name,
        "distances": {
            "robinson_foulds": movie_data.rfd_list,
            "weighted_robinson_foulds": movie_data.weighted_robinson_foulds_distance_list,
            "semantics": {
                "robinson_foulds": {
                    "topology": "rooted_clades",
                    "normalization": "symmetric_difference_over_union",
                    "scope": "adjacent_processed_input_trees",
                },
                "weighted_robinson_foulds": {
                    "topology": "rooted_clades",
                    "includes_branch_lengths": True,
                    "includes_terminal_and_root_splits": True,
                    "scope": "adjacent_processed_input_trees",
                },
            },
        },
    }


def create_empty_movie_data(filename: str) -> MovieData:
    """Create empty MovieData for failed processing scenarios."""
    return MovieData(
        interpolated_trees=[],
        tree_metadata=[],
        rfd_list=[],
        weighted_robinson_foulds_distance_list=[],
        sorted_leaves=[],
        tree_pair_solutions={},
        pivot_edge_tracking=[],
        subtree_highlight_tracking=[],
        file_name=filename,
        window_size=1,
        window_step_size=1,
        msa_dict=None,
        pair_interpolation_ranges=[],
    )


# =============================================================================
# Pivot Edge Tracking
# =============================================================================


def _derive_pivot_edge_tracking_from_events(
    processed_tree_metadata: List[TreeMetadataType],
    events_by_pair: Dict[str, List[Dict[str, Any]]],
) -> List[Optional[List[int]]]:
    """Derive per-tree pivot edge tracking aligned to metadata indices."""
    tracking: List[Optional[List[int]]] = [None for _ in processed_tree_metadata]
    first_global_for_pair: Dict[str, int] = {}

    for idx, meta in enumerate(processed_tree_metadata):
        pair_key = meta.get("tree_pair_key")
        step = meta.get("step_in_pair")
        if pair_key and step == 1 and pair_key not in first_global_for_pair:
            first_global_for_pair[pair_key] = idx

    for pair_key, events in events_by_pair.items():
        start_global = first_global_for_pair.get(pair_key)
        if start_global is None:
            continue
        for ev in events:
            step_start, step_end = ev.get("step_range", [0, -1])
            split = ev.get("split")
            if split is None:
                continue
            for local_step in range(step_start, step_end + 1):
                idx = start_global + local_step
                if 0 <= idx < len(tracking):
                    meta = processed_tree_metadata[idx]
                    if meta.get("tree_pair_key") is None:
                        continue
                    if meta.get("frame_type") == "input_tree":
                        continue
                    tracking[idx] = split

    return tracking


# =============================================================================
# Split Change Events
# =============================================================================


def _extract_split_change_events_from_solutions(
    tree_pair_solutions: Dict[str, TreePairSolution],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract and serialize split_change_events per pair."""
    events: Dict[str, List[Dict[str, Any]]] = {}

    for pair_key, solution in tree_pair_solutions.items():
        pair_events = solution.get("split_change_events", [])
        serialized_events: List[Dict[str, Any]] = []
        for ev in pair_events:
            serialized_events.append(
                {
                    "split": serialize_partition_to_indices(ev["split"]),
                    "step_range": list(ev["step_range"]),
                }
            )
        events[pair_key] = serialized_events

    return events


# =============================================================================
# Timeline Building
# =============================================================================


def _build_split_change_timeline(
    tree_metadata: List[TreeMetadataType],
    tree_pair_solutions: Dict[str, TreePairSolution],
) -> List[Dict[str, Any]]:
    """Build a global timeline with originals, split events, and explicit gaps."""
    timeline: List[Dict[str, Any]] = []
    first_global_for_pair, originals = _index_timeline_anchors(tree_metadata)
    per_pair_events = _extract_split_change_events_from_solutions(tree_pair_solutions)

    original_tree_count = len(originals)
    for i in range(original_tree_count):
        _append_original_entry(timeline, i, originals)
        if i < original_tree_count - 1:
            pair_key = f"pair_{i}_{i + 1}"
            events = per_pair_events.get(pair_key, [])
            start_global = first_global_for_pair.get(pair_key)
            _append_pair_events(timeline, pair_key, events, start_global)

    return timeline


def _index_timeline_anchors(
    tree_metadata: List[TreeMetadataType],
) -> tuple[Dict[str, int], Dict[int, Dict[str, Any]]]:
    """Index first-step globals per pair and originals with their global indices."""
    first_global_for_pair: Dict[str, int] = {}
    originals: Dict[int, Dict[str, Any]] = {}
    orig_counter = 0

    for idx, meta in enumerate(tree_metadata):
        pair_key = meta.get("tree_pair_key")
        step = meta.get("step_in_pair")
        if pair_key and step == 1 and pair_key not in first_global_for_pair:
            first_global_for_pair[pair_key] = idx
        if pair_key is None:
            originals[orig_counter] = {
                "global_index": idx,
                "name": "",
            }
            orig_counter += 1

    return first_global_for_pair, originals


def _append_original_entry(
    timeline: List[Dict[str, Any]],
    tree_index: int,
    originals: Dict[int, Dict[str, Any]],
) -> None:
    """Append the original tree entry to the timeline if available."""
    if tree_index not in originals:
        return
    orig = originals[tree_index]
    timeline.append(
        {
            "type": "original",
            "tree_index": tree_index,
            "global_index": orig["global_index"],
            "name": orig.get("name", ""),
        }
    )


def _append_pair_events(
    timeline: List[Dict[str, Any]],
    pair_key: str,
    events: List[Dict[str, Any]],
    start_global: Optional[int],
) -> None:
    """Append split events for one pair, computing global ranges from the start index."""
    if not events:
        return
    for ev in events:
        step_start, step_end = ev.get("step_range", [0, -1])
        g_start = None if start_global is None else start_global + step_start
        g_end = None if start_global is None else start_global + step_end
        timeline.append(
            {
                "type": "split_event",
                "pair_key": pair_key,
                "split": ev.get("split"),
                "step_range_local": [step_start, step_end],
                "step_range_global": [g_start, g_end],
            }
        )


# =============================================================================
# Serialization Helpers
# =============================================================================


def _serialize_tree_pair_solutions(
    tree_pair_solutions: Dict[str, TreePairSolution],
) -> Dict[str, Dict[str, Any]]:
    """Convert TreePairSolution objects to a JSON-serializable dict."""
    serialized: Dict[str, Dict[str, Any]] = {}

    for pair_key, solution in tree_pair_solutions.items():
        item: Dict[str, Any] = {
            "affected_subtrees_by_split": _serialize_affected_subtrees_by_split(
                solution
            ),
            "attachment_edges_by_split": _serialize_attachment_edges_by_split(
                solution.get("attachment_edges_by_split", {}),
            ),
        }

        if "spr_move_events" in solution:
            item["spr_move_events"] = _serialize_spr_move_events(
                solution["spr_move_events"]
            )

        serialized[pair_key] = item

    return serialized


def _serialize_affected_subtrees_by_split(
    solution: TreePairSolution,
) -> Dict[str, Any]:
    affected_subtrees = solution["affected_subtrees_by_split"]
    wrapped_subtrees = {pivot: [parts] for pivot, parts in affected_subtrees.items()}
    return serialize_partition_dict_to_indices(wrapped_subtrees)


def _serialize_attachment_edges_by_split(
    attachment_edges_by_split: Dict[Any, Dict[Any, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    serialized: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for pivot, mover_entries in attachment_edges_by_split.items():
        serialized[_partition_key(pivot)] = {
            _partition_key(mover): {
                "source": serialize_partition_to_indices(edges["source"]) or [],
                "destination": serialize_partition_to_indices(edges["destination"])
                or [],
            }
            for mover, edges in mover_entries.items()
        }

    return serialized


def _partition_key(partition: Any) -> str:
    return str(serialize_partition_to_indices(partition))


def _serialize_spr_path(path: List[SprPathSegment]) -> List[Dict[str, Any]]:
    return [
        {
            "split": serialize_partition_to_indices(segment["split"]),
            "branch_length": segment["branch_length"],
        }
        for segment in path
    ]


def _serialize_spr_move_events(events: List[SprMoveEvent]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for event in events:
        serialized.append(
            {
                "pivot_edge": serialize_partition_to_indices(event["pivot_edge"]),
                "driver_subtree": serialize_partition_to_indices(
                    event["driver_subtree"]
                ),
                "highlight_group": [
                    serialize_partition_to_indices(partition)
                    for partition in event["highlight_group"]
                ],
                "step_range": list(event["step_range"]),
                "collapse_path": _serialize_spr_path(event["collapse_path"]),
                "expand_path": _serialize_spr_path(event["expand_path"]),
                "collapse_hops": event["collapse_hops"],
                "expand_hops": event["expand_hops"],
                "total_hops": event["total_hops"],
                "collapse_branch_length": event["collapse_branch_length"],
                "expand_branch_length": event["expand_branch_length"],
                "total_branch_length": event["total_branch_length"],
            }
        )
    return serialized


def _process_tree_metadata(
    tree_metadata: List[TreeMetadataType],
) -> List[TreeMetadataType]:
    """Process tree metadata to ensure JSON serialization."""
    processed_metadata: List[TreeMetadataType] = []

    for meta in tree_metadata:
        processed_metadata.append(
            TreeMetadataType(
                tree_pair_key=meta.get("tree_pair_key"),
                step_in_pair=meta.get("step_in_pair"),
                source_tree_global_index=meta.get("source_tree_global_index"),
                frame_type=meta.get("frame_type"),
                state_semantics=meta.get("state_semantics"),
                is_observed_input=meta.get("is_observed_input"),
            )
        )

    return processed_metadata
