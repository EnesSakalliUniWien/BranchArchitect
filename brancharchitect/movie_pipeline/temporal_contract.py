"""Build normalized temporal rows for frontend playback and analytics."""

from __future__ import annotations

from itertools import groupby
from typing import Any, Dict, List, Mapping, Optional, Sequence

from brancharchitect.elements.partition import Partition, partition_size_bitmask_key
from brancharchitect.movie_pipeline.types import PAIR_METRIC_SEMANTICS
from brancharchitect.tree_interpolation.types import (
    SprPathSegment,
    TreeInterpolationSequence,
    AttachmentEdges,
)


def build_temporal_contract(
    sequence: TreeInterpolationSequence,
    robinson_foulds_distances: List[float],
    weighted_robinson_foulds_distances: List[float],
) -> Dict[str, Any]:
    """Build the primary temporal rows emitted by process_trees()."""
    original_tree_global_indices = sequence.get_original_tree_indices()
    frames = _build_frame_rows(
        len(sequence.interpolated_trees),
        original_tree_global_indices,
    )
    pairs = _build_pair_rows(sequence, original_tree_global_indices, frames)

    return {
        "frames": frames,
        "pairs": pairs,
        "temporal_events": _build_temporal_event_rows(sequence, pairs),
        "pair_metrics": _build_pair_metric_rows(
            pairs,
            robinson_foulds_distances,
            weighted_robinson_foulds_distances,
        ),
    }


def _build_frame_rows(
    frame_count: int,
    original_tree_global_indices: List[int],
) -> List[Dict[str, Any]]:
    """Build canonical frame rows parallel to the interpolated tree stream."""
    input_tree_index_by_frame = {
        frame_index: input_tree_index
        for input_tree_index, frame_index in enumerate(original_tree_global_indices)
    }
    rows = [
        {
            "frame_index": frame_index,
            "frame_type": "interpolation_frame",
            "state_semantics": "algorithmic_intermediate",
            "is_observed_input": False,
            "input_tree_index": None,
            "pair_id": None,
            "pair_ordinal": None,
            "local_step_index": None,
            "source_frame_index": None,
            "target_frame_index": None,
        }
        for frame_index in range(frame_count)
    ]

    for frame_index, input_tree_index in input_tree_index_by_frame.items():
        rows[frame_index] = {
            **rows[frame_index],
            "frame_type": "input_tree",
            "state_semantics": "processed_input_tree",
            "is_observed_input": True,
            "input_tree_index": input_tree_index,
        }

    originals = sorted(original_tree_global_indices)
    for pair_ordinal in range(len(originals) - 1):
        source_frame_index = originals[pair_ordinal]
        target_frame_index = originals[pair_ordinal + 1]
        pair_id = f"pair_{pair_ordinal}_{pair_ordinal + 1}"
        for frame_index in range(source_frame_index + 1, target_frame_index):
            rows[frame_index] = {
                **rows[frame_index],
                "pair_id": pair_id,
                "pair_ordinal": pair_ordinal,
                "local_step_index": frame_index - source_frame_index - 1,
                "source_frame_index": source_frame_index,
                "target_frame_index": target_frame_index,
            }

    return rows


def _build_pair_rows(
    sequence: TreeInterpolationSequence,
    original_tree_global_indices: List[int],
    frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build canonical pair rows with input anchors and movement context."""
    input_tree_index_by_frame = _input_tree_index_by_frame(frames)
    pair_ranges = sequence.get_pair_ranges(original_tree_global_indices)
    rows: List[Dict[str, Any]] = []

    for pair_ordinal, (source_frame_index, target_frame_index) in enumerate(
        pair_ranges
    ):
        pair_id = _pair_id(pair_ordinal)
        source_input_tree_index = input_tree_index_by_frame[source_frame_index]
        target_input_tree_index = input_tree_index_by_frame[target_frame_index]

        first_generated = source_frame_index + 1
        last_generated = target_frame_index - 1
        generated_frame_range = (
            [first_generated, last_generated]
            if first_generated <= last_generated
            else None
        )
        pivot_order = _ordered_pair_pivots(
            sequence.active_pivot_edges[source_frame_index + 1 : target_frame_index],
            (
                sequence.affected_subtrees_by_split_list[pair_ordinal],
                sequence.attachment_edge_maps[pair_ordinal],
            ),
        )

        rows.append(
            {
                "pair_id": pair_id,
                "pair_ordinal": pair_ordinal,
                "source_input_tree_index": source_input_tree_index,
                "target_input_tree_index": target_input_tree_index,
                "source_frame_index": source_frame_index,
                "target_frame_index": target_frame_index,
                "generated_frame_range": generated_frame_range,
                "solution": {
                    "affected_subtrees_by_split": _serialize_affected_subtrees_by_split(
                        sequence.affected_subtrees_by_split_list[pair_ordinal],
                        pivot_order,
                    ),
                    "attachment_edges_by_split": _serialize_attachment_edges_by_split(
                        sequence.attachment_edge_maps[pair_ordinal],
                        pivot_order,
                    ),
                },
            }
        )

    return rows


def _build_temporal_event_rows(
    sequence: TreeInterpolationSequence,
    pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build canonical temporal event rows with global frame ranges."""
    rows: List[Dict[str, Any]] = []

    for pair in pairs:
        pair_id = pair["pair_id"]
        pair_ordinal = pair["pair_ordinal"]
        source_frame_index = pair["source_frame_index"]
        target_frame_index = pair["target_frame_index"]
        pivot_sequence = sequence.active_pivot_edges[
            source_frame_index + 1 : target_frame_index
        ]
        split_sequence: List[Partition] = [
            pivot for pivot in pivot_sequence if pivot is not None
        ]

        for event_index, (split, local_step_range) in enumerate(
            _iter_split_change_ranges(split_sequence)
        ):
            rows.append(
                {
                    "event_id": f"{pair_id}:split:{event_index}",
                    "event_type": "split_change",
                    "pair_id": pair_id,
                    "pair_ordinal": pair_ordinal,
                    "local_step_range": local_step_range,
                    "frame_range": _local_range_to_frame_range(
                        source_frame_index,
                        local_step_range,
                    ),
                    "split": _serialize_partition_to_indices(split),
                }
            )

        for event_index, event in enumerate(
            sequence.spr_move_events_list[pair_ordinal]
        ):
            local_step_range = list(event["step_range"])
            rows.append(
                {
                    "event_id": f"{pair_id}:spr:{event_index}",
                    "event_type": "spr_move",
                    "pair_id": pair_id,
                    "pair_ordinal": pair_ordinal,
                    "local_step_range": local_step_range,
                    "frame_range": _local_range_to_frame_range(
                        source_frame_index,
                        local_step_range,
                    ),
                    "pivot_edge": _serialize_partition_to_indices(event["pivot_edge"]),
                    "driver_subtree": _serialize_partition_to_indices(
                        event["driver_subtree"]
                    ),
                    "highlight_group": [
                        _serialize_partition_to_indices(partition)
                        for partition in event["highlight_group"]
                    ],
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

    return rows


def _build_pair_metric_rows(
    pairs: List[Dict[str, Any]],
    robinson_foulds_distances: List[float],
    weighted_robinson_foulds_distance_list: List[float],
) -> Dict[str, Any]:
    if len(robinson_foulds_distances) != len(pairs):
        raise ValueError("robinson_foulds_distances must contain one value per pair")
    if len(weighted_robinson_foulds_distance_list) != len(pairs):
        raise ValueError(
            "weighted_robinson_foulds_distance_list must contain one value per pair"
        )

    return {
        "rows": [
            {
                "pair_id": pair["pair_id"],
                "pair_ordinal": pair["pair_ordinal"],
                "robinson_foulds": robinson_foulds_distances[pair["pair_ordinal"]],
                "weighted_robinson_foulds": weighted_robinson_foulds_distance_list[
                    pair["pair_ordinal"]
                ],
            }
            for pair in pairs
        ],
        "semantics": PAIR_METRIC_SEMANTICS,
    }


def _local_range_to_frame_range(
    source_frame_index: int,
    local_step_range: List[int],
) -> List[int]:
    return [
        source_frame_index + local_step_range[0] + 1,
        source_frame_index + local_step_range[1] + 1,
    ]


def _iter_split_change_ranges(
    split_sequence: Sequence[Partition],
) -> List[tuple[Partition, List[int]]]:
    """Aggregate contiguous changing-split occurrences within one pair."""
    ranges: List[tuple[Partition, List[int]]] = []
    start_idx = 0

    for split, group in groupby(split_sequence):
        group_size = sum(1 for _ in group)
        ranges.append((split, [start_idx, start_idx + group_size - 1]))
        start_idx += group_size

    return ranges


def _pair_id(pair_ordinal: int) -> str:
    return f"pair_{pair_ordinal}_{pair_ordinal + 1}"


def _input_tree_index_by_frame(
    frames: List[Dict[str, Any]],
) -> Dict[int, int]:
    """Map input-frame indices to their adjacent input-tree ordinal."""
    return {
        frame["frame_index"]: frame["input_tree_index"]
        for frame in frames
        if frame["frame_type"] == "input_tree"
    }


def _ordered_pair_pivots(
    active_pivot_edges: Sequence[Optional[Partition]],
    partition_maps: Sequence[Mapping[Partition, Any]],
) -> List[Partition]:
    """Return pivot keys in execution order, followed by deterministic leftovers."""
    ordered: List[Partition] = []
    seen_bitmasks: set[int] = set()

    for pivot in active_pivot_edges:
        if pivot is None or pivot.bitmask in seen_bitmasks:
            continue
        if any(pivot in partition_map for partition_map in partition_maps):
            ordered.append(pivot)
            seen_bitmasks.add(pivot.bitmask)

    leftovers: List[Partition] = []
    for partition_map in partition_maps:
        for pivot in partition_map:
            if pivot.bitmask in seen_bitmasks:
                continue
            leftovers.append(pivot)
            seen_bitmasks.add(pivot.bitmask)

    return ordered + sorted(leftovers, key=partition_size_bitmask_key)


def _order_partition_dict(
    partition_dict: Mapping[Partition, Any],
    pivot_order: Sequence[Partition],
) -> Dict[Partition, Any]:
    ordered: Dict[Partition, Any] = {}
    for pivot in pivot_order:
        if pivot in partition_dict:
            ordered[pivot] = partition_dict[pivot]

    for pivot in sorted(partition_dict, key=partition_size_bitmask_key):
        if pivot not in ordered:
            ordered[pivot] = partition_dict[pivot]

    return ordered


def _serialize_affected_subtrees_by_split(
    affected_subtrees: Dict[Partition, List[Partition]],
    pivot_order: Sequence[Partition],
) -> Dict[str, Any]:
    wrapped_subtrees = {
        pivot: [parts]
        for pivot, parts in _order_partition_dict(
            affected_subtrees,
            pivot_order,
        ).items()
    }
    return _serialize_partition_dict_to_indices(wrapped_subtrees)


def _serialize_attachment_edges_by_split(
    attachment_edges_by_split: Dict[Partition, Dict[Partition, AttachmentEdges]],
    pivot_order: Sequence[Partition],
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    serialized: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    ordered_attachment_edges = _order_partition_dict(
        attachment_edges_by_split,
        pivot_order,
    )
    for pivot, mover_entries in ordered_attachment_edges.items():
        serialized[_partition_key(pivot)] = {
            _partition_key(mover): {
                "source": _serialize_required_partition(edges["source"]),
                "destination": _serialize_required_partition(edges["destination"]),
            }
            for mover, edges in mover_entries.items()
        }

    return serialized


def _serialize_partition_dict_to_indices(
    partition_dict: Dict[Any, Any],
) -> Dict[str, Any]:
    def _serialize_value(value: Any) -> Any:
        if hasattr(value, "indices"):
            return _serialize_partition_to_indices(value)
        if isinstance(value, list):
            return [_serialize_value(item) for item in value]
        if isinstance(value, dict):
            return _serialize_partition_dict_to_indices(value)
        return value

    return {
        str(_serialize_partition_to_indices(key)): _serialize_value(value)
        for key, value in partition_dict.items()
    }


def _partition_key(partition: Any) -> str:
    return str(_serialize_required_partition(partition))


def _serialize_required_partition(partition: Any) -> List[int]:
    indices = _serialize_partition_to_indices(partition)
    if indices is None:
        raise ValueError("Expected Partition, got None")
    return indices


def _serialize_partition_to_indices(
    partition: Optional[Partition],
) -> Optional[List[int]]:
    return list(partition.indices) if partition is not None else None


def _serialize_spr_path(path: List[SprPathSegment]) -> List[Dict[str, Any]]:
    return [
        {
            "split": _serialize_partition_to_indices(segment["split"]),
            "branch_length": segment["branch_length"],
        }
        for segment in path
    ]
