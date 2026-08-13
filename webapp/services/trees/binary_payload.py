"""Binary container for the compact frontend tree payload.

The JSON payload holds one array per tree node, nested by child. On the shipped
datasets that is roughly 240 bytes of browser heap per node once parsed, and
norovirus carries 4,232,120 nodes, so the parsed object graph costs about a
gigabyte before anything is drawn.

This module packs the same information as flat typed arrays, one block per tree,
which the frontend can hold as ``ArrayBuffer`` views and expand one tree at a
time. Node order within a block is preorder, matching the order the nested form
recurses in, so the two encodings carry the same tree with no reordering.

Container layout, little-endian throughout::

    magic          4 bytes   b"PMB1"
    header_length  uint32
    header         header_length bytes of UTF-8 JSON, zero padded to 8 bytes
    body           one block per tree, each starting 8-byte aligned

Per-tree block, for ``n`` nodes and ``m`` annotation entries::

    length         float64 * n        branch length
    parent         int32   * n        parent node index, -1 for the root
    name_ref       uint32  * n        index into tree_name_definitions
    split_ref      uint32  * n        index into split_definitions
    ann_offset     uint32  * (n + 1)  CSR row offsets into the two arrays below
    ann_def        uint32  * m        index into annotation_definitions
    ann_value      uint32  * m        index into annotation_value_definitions

``length`` stays float64 because branch lengths feed distance and layout maths
downstream, where silently narrowing to float32 would change results. It is
placed first so the block's 8-byte alignment carries to it.

Annotation values are mixed strings, numbers, booleans and lists, so they cannot
live in a typed array. They are interned into ``annotation_value_definitions``
and referenced by index, which is the same trick the payload already uses for
names and splits, and it compresses well because support values repeat heavily.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Tuple

import numpy as np

MAGIC = b"PMB1"
PAYLOAD_FORMAT_VERSION = 3
_HEADER_ALIGNMENT = 8


def _align_up(value: int, alignment: int = _HEADER_ALIGNMENT) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def _value_key(value: Any) -> str:
    """Stable dictionary key for an annotation value.

    Keyed on the JSON encoding rather than the value so that ``True``, ``1`` and
    ``1.0`` stay distinct entries - Python would otherwise collapse them into one
    dictionary key and hand the frontend back the wrong type.
    """
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


class _AnnotationValueTable:
    def __init__(self) -> None:
        self.values: List[Any] = []
        self._index_by_key: Dict[str, int] = {}

    def intern(self, value: Any) -> int:
        key = _value_key(value)
        existing = self._index_by_key.get(key)
        if existing is not None:
            return existing
        index = len(self.values)
        self._index_by_key[key] = index
        self.values.append(value)
        return index


def _flatten_tree(
    node: List[Any], value_table: _AnnotationValueTable
) -> Tuple[
    List[float], List[int], List[int], List[int], List[int], List[int], List[int]
]:
    """Walk one compact tree into parallel preorder arrays."""
    lengths: List[float] = []
    parents: List[int] = []
    name_refs: List[int] = []
    split_refs: List[int] = []
    ann_offsets: List[int] = [0]
    ann_defs: List[int] = []
    ann_values: List[int] = []

    # Explicit stack rather than recursion: input trees get deep enough on large
    # datasets to reach the interpreter's recursion limit.
    stack: List[Tuple[List[Any], int]] = [(node, -1)]
    while stack:
        current, parent_index = stack.pop()
        index = len(lengths)

        lengths.append(float(current[0]))
        parents.append(parent_index)
        name_refs.append(int(current[1]))
        split_refs.append(int(current[2]))

        annotation_values = current[3]
        if annotation_values:
            for definition_index, value in annotation_values:
                ann_defs.append(int(definition_index))
                ann_values.append(value_table.intern(value))
        ann_offsets.append(len(ann_defs))

        children = current[4] or []
        # Reversed so popping yields the original child order, which keeps the
        # flat order identical to a plain preorder recursion.
        for child in reversed(children):
            stack.append((child, index))

    return lengths, parents, name_refs, split_refs, ann_offsets, ann_defs, ann_values


def pack_movie_payload(payload: Dict[str, Any]) -> bytes:
    """Pack a compact JSON movie payload into the PMB1 container.

    ``payload`` is the dict the JSON writer emits, including
    ``interpolated_trees`` in compact tuple form. Everything except the trees is
    copied into the header unchanged, so the two encodings stay in step without
    this module having to know the metadata contract.
    """
    trees = payload.get("interpolated_trees") or []
    metadata = {
        key: value for key, value in payload.items() if key != "interpolated_trees"
    }

    value_table = _AnnotationValueTable()
    blocks: List[bytes] = []
    directory: List[Dict[str, int]] = []
    body_offset = 0

    for tree in trees:
        (
            lengths,
            parents,
            name_refs,
            split_refs,
            ann_offsets,
            ann_defs,
            ann_values,
        ) = _flatten_tree(tree, value_table)

        block = b"".join(
            (
                np.asarray(lengths, dtype="<f8").tobytes(),
                np.asarray(parents, dtype="<i4").tobytes(),
                np.asarray(name_refs, dtype="<u4").tobytes(),
                np.asarray(split_refs, dtype="<u4").tobytes(),
                np.asarray(ann_offsets, dtype="<u4").tobytes(),
                np.asarray(ann_defs, dtype="<u4").tobytes(),
                np.asarray(ann_values, dtype="<u4").tobytes(),
            )
        )
        padding = _align_up(len(block)) - len(block)
        if padding:
            block += b"\x00" * padding

        directory.append(
            {
                "node_count": len(lengths),
                "annotation_count": len(ann_defs),
                "offset": body_offset,
            }
        )
        blocks.append(block)
        body_offset += len(block)

    header = {
        "payload_format_version": PAYLOAD_FORMAT_VERSION,
        "tree_count": len(trees),
        "annotation_value_definitions": value_table.values,
        "trees": directory,
        "metadata": metadata,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_padding = _align_up(len(header_bytes)) - len(header_bytes)

    return b"".join(
        (
            MAGIC,
            struct.pack("<I", len(header_bytes)),
            header_bytes,
            b"\x00" * header_padding,
            *blocks,
        )
    )


def _read_tree_block(
    buffer: bytes,
    body_start: int,
    entry: Dict[str, int],
    value_definitions: List[Any],
) -> List[Any]:
    node_count = entry["node_count"]
    annotation_count = entry["annotation_count"]
    cursor = body_start + entry["offset"]

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal cursor
        array = np.frombuffer(buffer, dtype=dtype, count=count, offset=cursor)
        cursor += array.nbytes
        return array

    lengths = take("<f8", node_count)
    parents = take("<i4", node_count)
    name_refs = take("<u4", node_count)
    split_refs = take("<u4", node_count)
    ann_offsets = take("<u4", node_count + 1)
    ann_defs = take("<u4", annotation_count)
    ann_values = take("<u4", annotation_count)

    nodes: List[List[Any]] = []
    for index in range(node_count):
        start = int(ann_offsets[index])
        end = int(ann_offsets[index + 1])
        annotation_values = (
            [
                [int(ann_defs[offset]), value_definitions[int(ann_values[offset])]]
                for offset in range(start, end)
            ]
            if end > start
            else None
        )
        nodes.append(
            [
                float(lengths[index]),
                int(name_refs[index]),
                int(split_refs[index]),
                annotation_values,
                [],
            ]
        )

    for index in range(node_count):
        parent_index = int(parents[index])
        if parent_index >= 0:
            nodes[parent_index][4].append(nodes[index])

    return nodes[0] if nodes else []


def unpack_movie_payload(buffer: bytes) -> Dict[str, Any]:
    """Rebuild the compact JSON movie payload from a PMB1 container.

    The inverse of :func:`pack_movie_payload`, used to prove round-trip equality
    in tests and to read a packed payload back from Python tooling.
    """
    if buffer[:4] != MAGIC:
        raise ValueError("payload is not a PMB1 container")

    (header_length,) = struct.unpack_from("<I", buffer, 4)
    header_start = 4 + 4
    header = json.loads(
        buffer[header_start : header_start + header_length].decode("utf-8")
    )

    version = header.get("payload_format_version")
    if version != PAYLOAD_FORMAT_VERSION:
        raise ValueError(f"unsupported payload_format_version {version!r}")

    body_start = header_start + _align_up(header_length)
    value_definitions = header["annotation_value_definitions"]
    trees = [
        _read_tree_block(buffer, body_start, entry, value_definitions)
        for entry in header["trees"]
    ]

    return {**header["metadata"], "interpolated_trees": trees}
