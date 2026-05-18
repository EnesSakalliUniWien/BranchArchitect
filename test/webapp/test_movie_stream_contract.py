import json

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
from webapp.services.trees.stream_contract import send_movie_stream


class PartitionStub:
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices


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


def test_movie_stream_contract_sends_metadata_chunks_and_complete_count() -> None:
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
    assert events[5][1] == {"data": {"tree_count": 2}}


def test_pipeline_frontend_metadata_is_aligned_with_serialized_trees() -> None:
    parsed = parse_newick("((A:1,B:1):1,C:1);(A:1,(B:1,C:1):1);")
    trees = [parsed] if isinstance(parsed, Node) else parsed

    result = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True, circular=True)
    ).process_trees(trees)
    sorted_leaves = [
        name
        for name, _ in sorted(trees[0].taxa_encoding.items(), key=lambda item: item[1])
    ]
    movie_data = build_movie_data_from_result(
        result,
        "example.nwk",
        {"inferred_window_size": 1, "inferred_step_size": 1, "msa_dict": None},
        sorted_leaves,
    )
    metadata = assemble_frontend_metadata(movie_data)

    tree_count = len(movie_data.interpolated_trees)
    assert tree_count > len(trees)
    assert len(metadata["tree_metadata"]) == tree_count
    assert len(metadata["pivot_edge_tracking"]) == tree_count
    assert len(metadata["subtree_highlight_tracking"]) == tree_count
    legacy_subtree_api_key = "subtree" + "_tracking"
    assert legacy_subtree_api_key not in metadata

    assert metadata["pair_interpolation_ranges"] == [[0, tree_count - 1]]
    assert metadata["split_change_timeline"][0] == {
        "type": "original",
        "tree_index": 0,
        "global_index": 0,
        "name": "",
    }
    assert metadata["split_change_timeline"][-1] == {
        "type": "original",
        "tree_index": 1,
        "global_index": tree_count - 1,
        "name": "",
    }

    for start, end in metadata["pair_interpolation_ranges"]:
        assert 0 <= start < end < tree_count


def test_movie_metadata_contract_has_no_top_level_split_change_events() -> None:
    movie_data = create_empty_movie_data("empty.nwk")

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert "split_change_events" not in metadata_payload


def test_movie_metadata_contract_keeps_pair_split_change_events_private() -> None:
    pivot = PartitionStub([1])
    movie_data = MovieData(
        interpolated_trees=[],
        tree_metadata=[
            {
                "tree_pair_key": None,
                "step_in_pair": None,
                "source_tree_global_index": None,
            },
            {
                "tree_pair_key": "pair_0_1",
                "step_in_pair": 1,
                "source_tree_global_index": 0,
            },
            {
                "tree_pair_key": None,
                "step_in_pair": None,
                "source_tree_global_index": None,
            },
        ],
        rfd_list=[],
        weighted_robinson_foulds_distance_list=[],
        sorted_leaves=[],
        tree_pair_solutions={
            "pair_0_1": {
                "affected_subtrees_by_split": {},
                "attachment_edges_by_split": {},
                "split_change_events": [
                    {
                        "split": pivot,
                        "step_range": (0, 0),
                    }
                ],
                "spr_move_events": [],
            }
        },
        pivot_edge_tracking=[],
        subtree_highlight_tracking=[],
        file_name="example.nwk",
        window_size=1,
        window_step_size=1,
        msa_dict=None,
        pair_interpolation_ranges=[],
    )

    metadata_payload = assemble_frontend_metadata(movie_data)

    pair_payload = metadata_payload["tree_pair_solutions"]["pair_0_1"]
    assert "split_change_events" not in pair_payload
    assert pair_payload == {
        "affected_subtrees_by_split": {},
        "attachment_edges_by_split": {},
        "spr_move_events": [],
    }
    assert metadata_payload["split_change_timeline"] == [
        {"type": "original", "tree_index": 0, "global_index": 0, "name": ""},
        {
            "type": "split_event",
            "pair_key": "pair_0_1",
            "split": [1],
            "step_range_local": [0, 0],
            "step_range_global": [1, 1],
        },
        {"type": "original", "tree_index": 1, "global_index": 2, "name": ""},
    ]


def test_movie_metadata_contract_has_no_tree_count() -> None:
    movie_data = create_empty_movie_data("empty.nwk")

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert "tree_count" not in metadata_payload
