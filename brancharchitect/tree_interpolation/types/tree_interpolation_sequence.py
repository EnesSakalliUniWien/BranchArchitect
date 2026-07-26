"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from .interpolation_movement import (
    AttachmentEdges,
    SprMoveEvent,
)

MappingDict = dict[Partition, dict[Partition, Partition]]
AttachmentEdgeMap = dict[Partition, dict[Partition, AttachmentEdges]]
JumpingSolutions = dict[Partition, list[Partition]]


def _empty_node_list() -> list[Node]:
    return []


def _empty_attachment_edge_maps() -> list[AttachmentEdgeMap]:
    return []


def _empty_partition_list() -> list[Optional[Partition]]:
    return []


def _empty_int_list() -> list[int]:
    return []


def _empty_jumping_solutions() -> list[JumpingSolutions]:
    return []


def _empty_spr_move_events() -> list[list[SprMoveEvent]]:
    return []


@dataclass
class TreeInterpolationSequence:
    """
    Comprehensive result structure from sequential lattice-based tree interpolation.

    This dataclass encapsulates all data generated during the interpolation of a sequence
    of phylogenetic trees. It replaces complex tuple returns with a clear, structured
    format that groups related data logically and provides convenient access methods.

    Core Structure:
    - For N input trees, emits each input tree once as a delimiter plus generated
      interpolation frames between delimiters
    - Each tree pair (Ti, Ti+1) produces 0 to many interpolation trees depending on pivot edges found
    - If Ti and Ti+1 are identical: 0 pivot edges found → 0 interpolation trees generated
    - If Ti and Ti+1 differ: generated frames exclude the exact destination
      endpoint because that state is represented by the next input delimiter

    Active Changing Split Tracking:
    - Original trees: None (no active changing split applied)
    - Interpolated trees: Specific Partition representing the active changing split being processed

    Attributes:
        interpolated_trees: Complete sequence of all trees (originals + interpolated)
        attachment_edge_maps: Source/destination attachment edges for each tree pair
            (outer key = pivot edge, inner key = moved subtree partition)
        active_pivot_edges: Active pivot edge applied for each tree (None for originals)
        pair_interpolated_tree_counts: Total interpolated trees generated per pair
        affected_subtrees_by_split_list: Affected subtrees grouped by active split per pair
        # distances removed

    Example:
        # For 3 input trees where T0≠T1 (2 s-edges found), T1=T2 (0 s-edges found):
        # Tree sequence: T0, [10 interpolated], T1, T2
        # Total trees: 3 + 10 + 0 = 13 trees (NOT 28!)

        from brancharchitect.tree_interpolation.sequential_interpolation import SequentialInterpolationBuilder
        result = SequentialInterpolationBuilder().build([tree1, tree2, tree3])
        # len(result.interpolated_trees) -> 13 (conditional!)
        # result.pair_interpolated_tree_counts -> [10, 0]
    """

    # Core interpolation results
    interpolated_trees: list[Node] = field(default_factory=_empty_node_list)
    attachment_edge_maps: list[AttachmentEdgeMap] = field(
        default_factory=_empty_attachment_edge_maps
    )
    active_pivot_edges: list[Optional[Partition]] = field(
        default_factory=_empty_partition_list
    )
    # Parallel to active_pivot_edges: None for originals, active mover
    # highlight groups for interpolated frames.
    current_subtree_highlights: list[Optional[list[Partition]]] = field(
        default_factory=list
    )
    pair_interpolated_tree_counts: list[int] = field(default_factory=_empty_int_list)
    affected_subtrees_by_split_list: list[JumpingSolutions] = field(
        default_factory=_empty_jumping_solutions
    )
    spr_move_events_list: list[list[SprMoveEvent]] = field(
        default_factory=_empty_spr_move_events
    )

    def get_pair_ranges(self, original_tree_indices: list[int]) -> list[list[int]]:
        """Compute source/destination delimiter ranges [start, end] for each pair."""
        pair_count = len(self.affected_subtrees_by_split_list)
        if len(original_tree_indices) < pair_count + 1:
            raise IndexError(
                "Not enough original tree delimiters to key solutions "
                f"(have {len(original_tree_indices)}, need {pair_count + 1})"
            )
        bounded = original_tree_indices[: pair_count + 1]
        return [[bounded[i], bounded[i + 1]] for i in range(pair_count)]

    def get_original_tree_indices(self) -> list[int]:
        """
        Get global indices of original (non-interpolated) trees in the sequence.

        Returns:
            List of indices where active_pivot_edges[i] is None, indicating original trees
        """
        return [
            i
            for i, pivot_edge in enumerate(self.active_pivot_edges)
            if pivot_edge is None
        ]

    def get_interpolated_tree_indices(self) -> list[int]:
        """
        Get global indices of interpolated trees in the sequence.

        Returns:
            List of indices where active_pivot_edges[i] is not None, indicating interpolated trees
        """
        return [
            i
            for i, pivot_edge in enumerate(self.active_pivot_edges)
            if pivot_edge is not None
        ]


def build_attachment_edge_map(
    source_map: MappingDict,
    destination_map: MappingDict,
) -> AttachmentEdgeMap:
    """Combine source/destination projections into one attachment-edge relation."""
    _assert_same_keys(source_map, destination_map, "attachment edge pivots")

    attachment_edges: AttachmentEdgeMap = {}
    for pivot, source_entries in source_map.items():
        destination_entries = destination_map[pivot]
        _assert_same_keys(
            source_entries,
            destination_entries,
            f"attachment edge movers for {pivot}",
        )
        attachment_edges[pivot] = {
            mover: {
                "source": source_edge,
                "destination": destination_entries[mover],
            }
            for mover, source_edge in source_entries.items()
        }

    return attachment_edges


def _assert_same_keys(
    source: Mapping[Partition, object],
    destination: Mapping[Partition, object],
    field_name: str,
) -> None:
    if set(source.keys()) != set(destination.keys()):
        raise ValueError(f"{field_name} must have matching source/destination keys")
