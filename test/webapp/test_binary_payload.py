"""Round-trip contract for the PMB1 binary payload container."""

from __future__ import annotations

import json
import struct
from typing import Any, Dict

import pytest

from webapp.services.trees.binary_payload import (
    MAGIC,
    PAYLOAD_FORMAT_VERSION,
    pack_movie_payload,
    unpack_movie_payload,
)


def _leaf(
    length: float, name_ref: int, split_ref: int, annotations: Any = None
) -> list:
    return [length, name_ref, split_ref, annotations, []]


def _payload(trees: list) -> Dict[str, Any]:
    return {
        "interpolated_trees": trees,
        "annotation_definitions": [
            {
                "key": "support.iqtree.sh_alrt",
                "path": ["support", "iqtree", "sh_alrt"],
                "label": "SH-aLRT",
                "value_type": "number",
                "role": "branch_support",
            },
            {
                "key": "label.raw_internal",
                "path": ["label", "raw_internal"],
                "label": "Raw Internal Label",
                "value_type": "string",
                "role": "source_annotation",
            },
        ],
        "tree_name_definitions": ["", "A", "B", "C"],
        "split_definitions": [[0, 1, 2], [0], [1], [2]],
        "frames": [{"frame_index": 0}],
        "pairs": [],
        "temporal_events": [],
        "subtree_highlight_tracking": [None],
        "pair_metrics": {"rows": [], "semantics": {}},
        "msa": {"sequences": None, "window_size": 10, "step_size": 5},
        "file_name": "binary-payload.trees",
        "dataset_provenance": None,
    }


SIMPLE_TREE = [
    0.0,
    0,
    0,
    None,
    [_leaf(1.5, 1, 1), _leaf(2.25, 2, 2), _leaf(0.125, 3, 3)],
]

ANNOTATED_TREE = [
    0.0,
    0,
    0,
    [[0, 88.5], [1, "internal-1"]],
    [
        _leaf(1.5, 1, 1, [[0, 100.0]]),
        _leaf(2.25, 2, 2, [[1, "leaf-label"]]),
        _leaf(0.125, 3, 3),
    ],
]


def test_round_trip_preserves_a_plain_tree() -> None:
    payload = _payload([SIMPLE_TREE])
    assert unpack_movie_payload(pack_movie_payload(payload)) == payload


def test_round_trip_preserves_annotations_and_metadata() -> None:
    payload = _payload([ANNOTATED_TREE, SIMPLE_TREE, ANNOTATED_TREE])
    assert unpack_movie_payload(pack_movie_payload(payload)) == payload


def test_round_trip_preserves_deeply_nested_trees() -> None:
    # A caterpillar deep enough that a recursive encoder would risk the
    # interpreter recursion limit on real datasets.
    tree: list = _leaf(0.5, 1, 1)
    for _ in range(5000):
        tree = [0.25, 0, 0, None, [tree]]

    packed = pack_movie_payload(_payload([tree]))
    restored = unpack_movie_payload(packed)["interpolated_trees"][0]

    depth = 0
    cursor = restored
    while cursor[4]:
        depth += 1
        cursor = cursor[4][0]
    assert depth == 5000


def test_child_order_is_preserved() -> None:
    payload = _payload([SIMPLE_TREE])
    restored = unpack_movie_payload(pack_movie_payload(payload))["interpolated_trees"][
        0
    ]
    assert [child[1] for child in restored[4]] == [1, 2, 3]


def test_annotation_values_of_different_types_stay_distinct() -> None:
    # json.dumps keying keeps True, 1 and 1.0 apart, where a plain dict would
    # collapse them into one entry and hand back the wrong type.
    tree = [0.0, 0, 0, [[0, True], [1, 1]], [_leaf(1.0, 1, 1, [[0, 1.0]])]]
    restored = unpack_movie_payload(pack_movie_payload(_payload([tree])))
    root = restored["interpolated_trees"][0]

    assert root[3] == [[0, True], [1, 1]]
    assert root[3][0][1] is True
    assert root[4][0][3] == [[0, 1.0]]


def test_annotation_values_are_interned_once() -> None:
    repeated = [0.0, 0, 0, [[0, 100.0]], [_leaf(1.0, 1, 1, [[0, 100.0]])]]
    packed = pack_movie_payload(_payload([repeated, repeated]))

    (header_length,) = struct.unpack_from("<I", packed, 4)
    header = json.loads(packed[8 : 8 + header_length].decode("utf-8"))

    assert header["annotation_value_definitions"] == [100.0]


def test_header_declares_the_format_version_and_tree_directory() -> None:
    packed = pack_movie_payload(_payload([ANNOTATED_TREE]))
    assert packed[:4] == MAGIC

    (header_length,) = struct.unpack_from("<I", packed, 4)
    header = json.loads(packed[8 : 8 + header_length].decode("utf-8"))

    assert header["payload_format_version"] == PAYLOAD_FORMAT_VERSION
    assert header["tree_count"] == 1
    assert header["trees"] == [{"node_count": 4, "annotation_count": 4, "offset": 0}]
    assert "interpolated_trees" not in header["metadata"]


def test_empty_payload_round_trips() -> None:
    payload = _payload([])
    assert unpack_movie_payload(pack_movie_payload(payload)) == payload


def test_rejects_a_buffer_that_is_not_a_container() -> None:
    with pytest.raises(ValueError, match="not a PMB1 container"):
        unpack_movie_payload(b"NOPE" + b"\x00" * 32)


def test_rejects_an_unsupported_format_version() -> None:
    packed = bytearray(pack_movie_payload(_payload([SIMPLE_TREE])))
    (header_length,) = struct.unpack_from("<I", packed, 4)
    header = json.loads(packed[8 : 8 + header_length].decode("utf-8"))
    header["payload_format_version"] = 99

    rewritten = json.dumps(header, separators=(",", ":")).encode("utf-8")
    tampered = (
        MAGIC
        + struct.pack("<I", len(rewritten))
        + rewritten
        + bytes(packed[8 + header_length :])
    )

    with pytest.raises(ValueError, match="unsupported payload_format_version"):
        unpack_movie_payload(tampered)
