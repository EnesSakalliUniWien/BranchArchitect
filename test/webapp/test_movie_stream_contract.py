import json
from typing import Any

from flask import Flask

from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from webapp.services.sse.channels import ProgressChannel
from webapp.services.trees.frontend_builder import (
    assemble_frontend_metadata,
    build_movie_data_from_result,
    create_empty_movie_data,
)
from webapp.services.trees.movie_data import MovieData
from webapp.services.trees.processing import handle_tree_content_streaming
from webapp.services.trees.stream_contract import send_movie_stream


class CapturingLog:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[object, ...]]] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append((message, args))


def _parse_sse_message(message: str) -> tuple[str | None, object]:
    event = None
    data_lines = []

    for line in message.strip().splitlines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))

    return event, json.loads("\n".join(data_lines))


def _find_split(
    metadata: dict[str, Any], node_dict: Any, split: list[int]
) -> Any | None:
    if isinstance(node_dict, list):
        split_indices = metadata["split_definitions"][node_dict[2]]
        children = node_dict[4]
    else:
        split_indices = node_dict.get("split_indices")
        if split_indices is None and "split_ref" in node_dict:
            split_indices = metadata["split_definitions"][node_dict["split_ref"]]
        children = node_dict["children"]
    if split_indices == split:
        return node_dict
    assert isinstance(children, list)
    for child in children:
        found = _find_split(metadata, child, split)
        if found is not None:
            return found
    return None


def _annotation_fields(
    metadata: dict[str, Any], node_dict: Any
) -> dict[str, dict[str, Any]]:
    definitions = metadata.get("annotation_definitions", [])
    annotation_values = (
        node_dict[3]
        if isinstance(node_dict, list)
        else node_dict.get("annotation_values", [])
    )
    fields: dict[str, dict[str, Any]] = {}
    for definition_index, value in annotation_values or []:
        definition = definitions[definition_index]
        key = definition["key"]
        fields[key] = {
            field_key: field_value
            for field_key, field_value in definition.items()
            if field_key != "key"
        }
        fields[key]["value"] = value
    return fields


def test_movie_stream_contract_sends_metadata_chunks_and_empty_complete_event() -> None:
    channel = ProgressChannel()
    metadata = {"file_name": "example.nwk"}
    trees = [{"name": "a"}, {"name": "b"}]

    send_movie_stream(channel, metadata, trees, CapturingLog(), chunk_size=1)

    events = [_parse_sse_message(message) for message in channel.stream(timeout=0.01)]

    assert [event for event, _payload in events] == [
        "progress",
        "metadata",
        "trees_chunk",
        "trees_chunk",
        "progress",
        "complete",
    ]
    assert events[1][1] == {"metadata": metadata}
    assert events[2][1] == {
        "trees": [{"name": "a"}],
        "start_index": 0,
        "end_index": 1,
        "total": 2,
    }
    assert events[3][1] == {
        "trees": [{"name": "b"}],
        "start_index": 1,
        "end_index": 2,
        "total": 2,
    }
    assert events[5][1] == {"data": None}


def test_pipeline_frontend_metadata_is_aligned_with_serialized_trees() -> None:
    parsed = parse_newick("((A:1,B:1):1,C:1);(A:1,(B:1,C:1):1);")
    trees = [parsed] if isinstance(parsed, Node) else parsed

    result = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True, circular=True)
    ).process_trees(trees)
    movie_data = build_movie_data_from_result(
        result,
        "example.nwk",
        {"inferred_window_size": 1, "inferred_step_size": 1, "msa_dict": None},
    )
    metadata = assemble_frontend_metadata(movie_data)

    total_trees = len(movie_data.interpolated_trees)
    assert total_trees > len(trees)
    assert len(metadata["frames"]) == total_trees
    assert len(metadata["subtree_highlight_tracking"]) == total_trees
    legacy_subtree_api_key = "subtree" + "_tracking"
    assert legacy_subtree_api_key not in metadata

    assert metadata["pairs"] == [
        {
            "pair_id": "pair_0_1",
            "pair_ordinal": 0,
            "source_input_tree_index": 0,
            "target_input_tree_index": 1,
            "source_frame_index": 0,
            "target_frame_index": total_trees - 1,
            "generated_frame_range": [1, total_trees - 2],
            "solution": metadata["pairs"][0]["solution"],
        }
    ]
    assert metadata["frames"][0]["frame_type"] == "input_tree"
    assert metadata["frames"][0]["input_tree_index"] == 0
    assert metadata["frames"][-1]["frame_type"] == "input_tree"
    assert metadata["frames"][-1]["input_tree_index"] == 1
    assert metadata["frames"][1]["pair_id"] == "pair_0_1"
    assert metadata["frames"][1]["local_step_index"] == 0
    assert metadata["temporal_events"]
    assert metadata["temporal_events"][0]["event_type"] == "split_change"
    assert metadata["temporal_events"][0]["pair_id"] == "pair_0_1"
    assert metadata["temporal_events"][0]["frame_range"][0] >= 1


def test_iqtree_support_mode_reaches_streamed_tree_annotations() -> None:
    app = Flask(__name__)

    with app.app_context():
        metadata, trees = handle_tree_content_streaming(
            "((A:1,B:1)95:2,C:3);",
            filename="iqtree.nwk",
            iqtree_support_mode="ufboot",
        )

    ab_node = _find_split(metadata, trees[0], [0, 1])
    assert ab_node is not None
    fields = _annotation_fields(metadata, ab_node)

    assert "support.bootstrap.value" not in fields
    assert fields["support.iqtree.ufboot"]["value"] == 95.0
    assert fields["support.iqtree.ufboot"]["analysis"] == {
        "type": "tree_inference",
        "method": "iqtree",
        "mode": "ufboot",
    }


def test_streamed_tree_annotations_are_compacted_into_metadata_definitions() -> None:
    app = Flask(__name__)

    with app.app_context():
        metadata, trees = handle_tree_content_streaming(
            "((A:1,B:1)95:2,C:3);",
            filename="iqtree.nwk",
            iqtree_support_mode="ufboot",
        )

    definitions = metadata["annotation_definitions"]
    assert definitions == [
        {
            "key": "label.raw_internal",
            "path": ["label", "raw_internal"],
            "label": "Raw Internal Label",
            "value_type": "string",
            "role": "source_annotation",
        },
        {
            "key": "support.iqtree.ufboot",
            "path": ["support", "iqtree", "ufboot"],
            "label": "UFBoot",
            "value_type": "number",
            "role": "branch_support",
            "unit": "percent",
            "analysis": {
                "type": "tree_inference",
                "method": "iqtree",
                "mode": "ufboot",
            },
        },
    ]

    ab_node = _find_split(metadata, trees[0], [0, 1])
    assert ab_node is not None
    assert "annotations" not in ab_node
    assert ab_node[3] == [[0, "95"], [1, 95.0]]


def test_streamed_tree_names_and_splits_are_compacted_into_metadata_definitions() -> (
    None
):
    app = Flask(__name__)

    with app.app_context():
        metadata, trees = handle_tree_content_streaming(
            "((A:1,B:1):2,C:3);",
            filename="compact.nwk",
        )

    assert metadata["tree_name_definitions"] == ["", "A", "B", "C"]
    assert metadata["split_definitions"] == [[0, 1, 2], [0, 1], [0], [1], [2]]
    assert trees[0] == [
        1,
        0,
        0,
        None,
        [
            [
                2.0,
                0,
                1,
                None,
                [
                    [1.0, 1, 2, None, []],
                    [1.0, 2, 3, None, []],
                ],
            ],
            [3.0, 3, 4, None, []],
        ],
    ]


def test_uploaded_tree_series_automatically_gets_split_frequency_support() -> None:
    app = Flask(__name__)

    with app.app_context():
        metadata, trees = handle_tree_content_streaming(
            "\n".join(
                [
                    "((A:1,B:1):1,(C:1,D:1):1);",
                    "((A:1,B:1):1,(C:1,D:1):1);",
                    "((A:1,C:1):1,(B:1,D:1):1);",
                ]
            ),
            filename="bootstrap_series.nwk",
            annotate_tree_series_support=True,
        )

    ab_node = _find_split(metadata, trees[0], [0, 1])
    assert ab_node is not None
    fields = _annotation_fields(metadata, ab_node)

    assert fields["support.bootstrap_rogue.frequency"]["value"] == 66.6667
    assert fields["support.bootstrap_rogue.replicate_count"]["value"] == 2.0
    assert fields["support.bootstrap_rogue.replicate_total"]["value"] == 3.0


def test_tree_series_split_frequency_does_not_overwrite_existing_support() -> None:
    app = Flask(__name__)

    with app.app_context():
        metadata, trees = handle_tree_content_streaming(
            "((A:1,B:1)95:1,(C:1,D:1):1);((A:1,B:1)95:1,(C:1,D:1):1);",
            filename="iqtree_support.nwk",
            iqtree_support_mode="ufboot",
            annotate_tree_series_support=True,
        )

    ab_node = _find_split(metadata, trees[0], [0, 1])
    assert ab_node is not None
    fields = _annotation_fields(metadata, ab_node)

    assert fields["support.iqtree.ufboot"]["value"] == 95.0
    assert "support.bootstrap_rogue.frequency" not in fields


def test_movie_data_uses_normalized_rows_as_primary_temporal_contract() -> None:
    parsed = parse_newick("((A:1,B:1):1,C:1);(A:1,(B:1,C:1):1);")
    trees = [parsed] if isinstance(parsed, Node) else parsed

    result = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True, circular=True)
    ).process_trees(trees)
    movie_data = build_movie_data_from_result(
        result,
        "example.nwk",
        {"inferred_window_size": 1, "inferred_step_size": 1, "msa_dict": None},
    )

    assert movie_data.frames
    assert movie_data.pairs
    assert movie_data.temporal_events
    assert movie_data.pair_metrics["rows"]
    assert not hasattr(movie_data, "tree_metadata")
    assert not hasattr(movie_data, "pair_interpolation_contexts")
    assert not hasattr(movie_data, "pair_interpolation_ranges")


def test_pipeline_result_uses_normalized_rows_as_primary_temporal_contract() -> None:
    parsed = parse_newick("((A:1,B:1):1,C:1);(A:1,(B:1,C:1):1);")
    trees = [parsed] if isinstance(parsed, Node) else parsed

    result = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True, circular=True)
    ).process_trees(trees)

    assert result["frames"]
    assert result["pairs"]
    assert result["temporal_events"]
    assert result["pair_metrics"]["rows"]
    assert "tree_metadata" not in result
    assert "pair_interpolation_contexts" not in result
    assert "pair_interpolation_ranges" not in result


def test_movie_metadata_contract_has_no_redundant_legacy_temporal_keys() -> None:
    movie_data = create_empty_movie_data("empty.nwk")

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert "split_change_events" not in metadata_payload
    assert "tree_metadata" not in metadata_payload
    assert "pair_interpolation_contexts" not in metadata_payload
    assert "split_change_timeline" not in metadata_payload
    assert "pair_interpolation_ranges" not in metadata_payload


def test_movie_metadata_contract_emits_normalized_pair_and_temporal_event_rows() -> (
    None
):
    frames = [
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
        },
        {
            "frame_index": 1,
            "frame_type": "interpolation_frame",
            "state_semantics": "algorithmic_intermediate",
            "is_observed_input": False,
            "input_tree_index": None,
            "pair_id": "pair_0_1",
            "pair_ordinal": 0,
            "local_step_index": 0,
            "source_frame_index": 0,
            "target_frame_index": 2,
        },
        {
            "frame_index": 2,
            "frame_type": "input_tree",
            "state_semantics": "processed_input_tree",
            "is_observed_input": True,
            "input_tree_index": 1,
            "pair_id": None,
            "pair_ordinal": None,
            "local_step_index": None,
            "source_frame_index": None,
            "target_frame_index": None,
        },
    ]
    pairs = [
        {
            "pair_id": "pair_0_1",
            "pair_ordinal": 0,
            "source_input_tree_index": 0,
            "target_input_tree_index": 1,
            "source_frame_index": 0,
            "target_frame_index": 2,
            "generated_frame_range": [1, 1],
            "solution": {
                "affected_subtrees_by_split": {},
                "attachment_edges_by_split": {},
            },
        }
    ]
    temporal_events = [
        {
            "event_id": "pair_0_1:split:0",
            "event_type": "split_change",
            "pair_id": "pair_0_1",
            "pair_ordinal": 0,
            "local_step_range": [0, 0],
            "frame_range": [1, 1],
            "split": [1],
        }
    ]
    pair_metrics = {
        "rows": [
            {
                "pair_id": "pair_0_1",
                "pair_ordinal": 0,
                "robinson_foulds": 0.25,
                "weighted_robinson_foulds": 1.25,
            }
        ],
        "semantics": {},
    }
    movie_data = MovieData(
        interpolated_trees=[],
        annotation_definitions=[],
        tree_name_definitions=[],
        split_definitions=[],
        frames=frames,
        pairs=pairs,
        temporal_events=temporal_events,
        pair_metrics=pair_metrics,
        subtree_highlight_tracking=[],
        file_name="example.nwk",
        window_size=1,
        window_step_size=1,
        msa_dict=None,
    )

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert metadata_payload["pairs"] == pairs
    assert metadata_payload["temporal_events"] == temporal_events
    assert metadata_payload["frames"] == frames
    assert metadata_payload["pair_metrics"] == pair_metrics


def test_movie_metadata_contract_has_no_tree_count() -> None:
    movie_data = create_empty_movie_data("empty.nwk")

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert "tree_count" not in metadata_payload


def test_movie_metadata_contract_has_exact_frontend_keys() -> None:
    movie_data = create_empty_movie_data("empty.nwk")

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert set(metadata_payload) == {
        "annotation_definitions",
        "tree_name_definitions",
        "split_definitions",
        "frames",
        "pairs",
        "temporal_events",
        "subtree_highlight_tracking",
        "pair_metrics",
        "msa",
        "file_name",
        "dataset_provenance",
    }
    assert metadata_payload["dataset_provenance"] is None
    assert set(metadata_payload["msa"]) == {"sequences", "window_size", "step_size"}
    assert set(metadata_payload["pair_metrics"]) == {
        "rows",
        "semantics",
    }
