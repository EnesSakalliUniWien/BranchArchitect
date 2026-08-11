"""
Builds the frontend-specific data structures from the backend processing result.

This module transforms the raw output from TreeInterpolationPipeline into
the exact format required by the frontend UI.

Key Responsibilities:
- Serialize tree objects for chunked streaming.
- Assemble the normalized metadata payload sent before streamed tree chunks.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from brancharchitect.movie_pipeline.types import (
    InterpolationResult,
    PAIR_METRIC_SEMANTICS,
)
from brancharchitect.tree import Node, build_branch_annotation_fields
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
    logger = logging.getLogger("webapp_pipeline")
    interpolated_trees = result["interpolated_trees"]

    t_compact_start = time.perf_counter()
    (
        compact_trees,
        annotation_definitions,
        tree_name_definitions,
        split_definitions,
    ) = compact_tree_payload(interpolated_trees)
    logger.info(
        "[PhaseTimer] compact_tree_payload count=%d %.3fs",
        len(interpolated_trees),
        time.perf_counter() - t_compact_start,
    )

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
    trees: List[Node],
) -> Tuple[List[Any], List[Dict[str, Any]], List[str], List[List[int]]]:
    """Build the compact frontend tree payload directly from ``Node`` objects.

    Walks each tree exactly once (rather than serializing to an intermediate
    dict tree first and then compacting that), moving repeated tree schemas
    and annotation schemas to top-level definitions. Tree nodes keep compact
    references for repeated values: ``name_ref`` for node names, ``split_ref``
    for split arrays, and ``annotation_values`` pairs for annotation fields.
    The frontend validator hydrates these back to the canonical tree node
    shape.
    """

    definition_indices_by_key: Dict[str, List[int]] = {}
    definitions: List[Dict[str, Any]] = []
    name_index_by_value: Dict[str, int] = {}
    tree_name_definitions: List[str] = []
    split_index_by_value: Dict[Tuple[int, ...], int] = {}
    split_definitions: List[List[int]] = []

    def definition_index(field_key: str, field: Dict[str, Any]) -> int:
        """
        Intern one definition per distinct schema, not per field key.

        A field key describes what an annotation is called; the rest of the
        field describes its schema, including ``value_type``. The same key can
        legitimately arrive with different schemas across trees - a metadata
        value that is a string on one tree and a number on another - so keying
        only on ``field_key`` handed the second value the first one's schema.
        Definitions stay addressed by index, so repeated keys are fine here.
        """
        definition = {key: value for key, value in field.items() if key != "value"}
        definition["key"] = field_key

        candidate_indices = definition_indices_by_key.get(field_key)
        if candidate_indices is None:
            candidate_indices = []
            definition_indices_by_key[field_key] = candidate_indices
        else:
            for candidate_index in candidate_indices:
                if definitions[candidate_index] == definition:
                    return candidate_index

        index = len(definitions)
        candidate_indices.append(index)
        definitions.append(definition)
        return index

    def name_index(name: str) -> int:
        existing_index = name_index_by_value.get(name)
        if existing_index is not None:
            return existing_index

        index = len(tree_name_definitions)
        name_index_by_value[name] = index
        tree_name_definitions.append(name)
        return index

    def split_index(split_indices: List[int]) -> int:
        split_tuple = tuple(int(index) for index in split_indices)

        existing_index = split_index_by_value.get(split_tuple)
        if existing_index is not None:
            return existing_index

        index = len(split_definitions)
        split_index_by_value[split_tuple] = index
        split_definitions.append(list(split_tuple))
        return index

    def compact_node(node: Node) -> List[Any]:
        if node.is_leaf():
            split_indices = list(node.split_indices.resolve_to_indices())
            name = node.name
        else:
            split_indices = list(node.split_indices.indices)
            name = ""

        annotation_fields = build_branch_annotation_fields(node)
        annotation_values = None
        if annotation_fields:
            annotation_values = [
                [definition_index(str(field_key), field), field["value"]]
                for field_key, field in annotation_fields.items()
                if isinstance(field, dict) and "value" in field
            ]

        return [
            node.length,
            name_index(name),
            split_index(split_indices),
            annotation_values,
            [compact_node(child) for child in node.children],
        ]

    return (
        [compact_node(tree) for tree in trees],
        definitions,
        tree_name_definitions,
        split_definitions,
    )
