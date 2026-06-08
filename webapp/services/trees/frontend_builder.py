"""
Builds the frontend-specific data structures from the backend processing result.

This module transforms the raw output from TreeInterpolationPipeline into
the exact format required by the frontend UI.

Key Responsibilities:
- Serialize tree objects for chunked streaming.
- Assemble the normalized metadata payload sent before streamed tree chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
    (
        compact_trees,
        annotation_definitions,
        tree_name_definitions,
        split_definitions,
    ) = compact_tree_payload(serialized_trees)

    return MovieData(
        interpolated_trees=compact_trees,
        annotation_definitions=annotation_definitions,
        tree_name_definitions=tree_name_definitions,
        split_definitions=split_definitions,
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
        "annotation_definitions": movie_data.annotation_definitions,
        "tree_name_definitions": movie_data.tree_name_definitions,
        "split_definitions": movie_data.split_definitions,
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
        "dataset_provenance": None,
    }


def create_empty_movie_data(filename: str) -> MovieData:
    """Create empty MovieData for failed processing scenarios."""
    return MovieData(
        interpolated_trees=[],
        annotation_definitions=[],
        tree_name_definitions=[],
        split_definitions=[],
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


def compact_tree_payload(
    trees: List[Dict[str, Any]],
) -> Tuple[List[Any], List[Dict[str, Any]], List[str], List[List[int]]]:
    """Move repeated tree schemas and annotation schemas to top-level definitions.

    Tree nodes keep compact references for repeated values:
    ``name_ref`` for node names, ``split_ref`` for split arrays, and
    ``annotation_values`` pairs for annotation fields. The frontend validator
    hydrates these back to the canonical tree node shape.
    """

    definition_index_by_key: Dict[str, int] = {}
    definitions: List[Dict[str, Any]] = []
    name_index_by_value: Dict[str, int] = {}
    tree_name_definitions: List[str] = []
    split_index_by_value: Dict[Tuple[int, ...], int] = {}
    split_definitions: List[List[int]] = []

    def definition_index(field_key: str, field: Dict[str, Any]) -> int:
        existing_index = definition_index_by_key.get(field_key)
        if existing_index is not None:
            return existing_index

        definition = {key: value for key, value in field.items() if key != "value"}
        definition["key"] = field_key
        index = len(definitions)
        definition_index_by_key[field_key] = index
        definitions.append(definition)
        return index

    def name_index(name: Any) -> int:
        name_value = name if isinstance(name, str) else ""
        existing_index = name_index_by_value.get(name_value)
        if existing_index is not None:
            return existing_index

        index = len(tree_name_definitions)
        name_index_by_value[name_value] = index
        tree_name_definitions.append(name_value)
        return index

    def split_index(split_indices: Any) -> int:
        if isinstance(split_indices, list):
            split_tuple = tuple(int(index) for index in split_indices)
        else:
            split_tuple = ()

        existing_index = split_index_by_value.get(split_tuple)
        if existing_index is not None:
            return existing_index

        index = len(split_definitions)
        split_index_by_value[split_tuple] = index
        split_definitions.append(list(split_tuple))
        return index

    def compact_node(node: Dict[str, Any]) -> List[Any]:
        annotations = node.get("annotations")
        fields = annotations.get("fields") if isinstance(annotations, dict) else None
        annotation_values = None
        if isinstance(fields, dict) and fields:
            annotation_values = []
            for field_key, field in fields.items():
                if not isinstance(field, dict) or "value" not in field:
                    continue
                annotation_values.append(
                    [definition_index(str(field_key), field), field["value"]]
                )

        node_name_ref = name_index(node.get("name"))
        node_split_ref = split_index(node.get("split_indices"))
        children = [
            compact_node(child)
            for child in node.get("children", [])
            if isinstance(child, dict)
        ]
        return [
            node.get("length", 0),
            node_name_ref,
            node_split_ref,
            annotation_values,
            children,
        ]

    return (
        [compact_node(tree) for tree in trees],
        definitions,
        tree_name_definitions,
        split_definitions,
    )


def compact_tree_annotations(
    trees: List[Dict[str, Any]],
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Backward-compatible helper for callers that only need annotation metadata."""

    compacted, annotation_definitions, _tree_names, _splits = compact_tree_payload(
        trees
    )
    return compacted, annotation_definitions
