import json

from webapp.services.sse.channels import ProgressChannel
from webapp.services.trees.frontend_builder import (
    assemble_frontend_metadata,
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
                "jumping_subtree_solutions": {},
                "solution_to_source_map": {},
                "solution_to_destination_map": {},
                "split_change_events": [
                    {
                        "split": pivot,
                        "step_range": (0, 0),
                    }
                ],
            }
        },
        pivot_edge_tracking=[],
        subtree_tracking=[],
        file_name="example.nwk",
        window_size=1,
        window_step_size=1,
        msa_dict=None,
        pair_interpolation_ranges=[],
    )

    metadata_payload = assemble_frontend_metadata(movie_data)

    assert "split_change_events" not in metadata_payload["tree_pair_solutions"]["pair_0_1"]
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
